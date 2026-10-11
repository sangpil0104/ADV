from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from .accident_group import validate_group
from .contracts import EditRequest, Source
from .errors import DataError
from .group_geometry import feature_grid, group_grid_geometry
from .segmentation import load_mask
from .storage import atomic_image, digest, file_hash, read_json, within


def grid_point(point, width, height):
    grid = feature_grid(width, height)
    return [round(point[i] / (width, height)[i] * grid[i]) for i in (0, 1)]


def candidate_id(source: Source, cfg: dict, attempt_index: int) -> str:
    suffix = digest([cfg["run"]["seed"], source.source_id, source.pixel_sha256])[:10]
    return f"{source.source_id[:80]}_{attempt_index:08d}_{suffix}"


def sample_edit(source: Source, segments: dict, cfg: dict, attempt_index: int, run_dir: Path) -> EditRequest:
    settings = cfg["sampler"]
    seed = int(digest([cfg["run"]["seed"], source.source_id, source.pixel_sha256, attempt_index])[:8], 16)
    rng = np.random.default_rng(seed)
    w, h = source.width, source.height
    group = segments.get("accident_group")
    mask = validate_group(group, segments["segments"], run_dir, (w, h))
    members = [load_mask(run_dir, m["mask_path"], m["mask_sha256"], (w, h)) for m in group["members"]]
    ys, xs = np.nonzero(mask)
    # This virtual centroid may be between vehicles or in a mask hole.
    start = [float(xs.mean()), float(ys.mean())]
    for _ in range(settings["geometry_attempts"]):
        delta = [
            int(rng.integers(-int(xs.min()), w - int(xs.max()))),
            int(rng.integers(-int(ys.min()), h - int(ys.max()))),
        ]
        try:
            geometry = group_grid_geometry(mask, delta, members)
        except DataError:
            continue
        break
    else:
        raise DataError("No valid nonidentity whole accident group translation; no partial-mask fallback")
    end = [start[i] + delta[i] for i in (0, 1)]
    eid = candidate_id(source, cfg, attempt_index)
    mask_path = f".staging/{eid}/edit_mask.png"
    atomic_image(within(run_dir, mask_path), Image.fromarray(mask.astype(np.uint8) * 255))
    source_prompt = cfg["generation"]["source_prompt"]
    if source.preserve_hint:
        source_prompt += " Source description: " + source.preserve_hint
    target_prompt = (
        f"{source_prompt} Move all {len(members)} selected accident vehicles together as one group "
        f"by {delta} pixels from {start} toward {end}, preserving their relative positions, sizes and contact. "
        f"{cfg['generation']['effective_augmentation_prompt']} "
        "Preserve the traffic accident and do not repair visible accident damage."
    )
    plan = EditRequest(
        source_id=source.source_id,
        edit_id=eid,
        attempt_index=attempt_index,
        seed=seed,
        sampler_version=settings["version"],
        operation="relocation",
        operation_params={"displacement_pixels": delta, "group_grid_geometry": geometry},
        region_mask_path=mask_path,
        source_point=start,
        target_point=end,
        anchor_point=None,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        mask_sha256=file_hash(within(run_dir, mask_path)),
        segment_id=group["group_id"],
        segment_mask_path=group["mask_path"],
        segment_mask_hash=group["mask_sha256"],
        range_fraction_of_image=int(mask.sum()) / (w * h),
        area_pixels=int(mask.sum()),
        augmentation_prompt=cfg["generation"].get("augmentation_prompt"),
        accident_group=group,
    )
    validate_plan(plan, source, run_dir)
    return plan


def validate_plan(plan: EditRequest, source: Source, run_dir: Path) -> None:
    if (
        plan.schema_version != "2.0"
        or plan.source_id != source.source_id
        or plan.operation != "relocation"
        or plan.sampler_version != "accident_group_v1"
        or plan.anchor_point is not None
    ):
        raise DataError("Invalid source, schema, or whole-group operation in edit plan")
    for point in (plan.source_point, plan.target_point):
        if len(point) != 2 or not all(type(x) in (int, float) and math.isfinite(x) for x in point):
            raise DataError("Invalid point")
        if not 0 <= point[0] < source.width or not 0 <= point[1] < source.height:
            raise DataError("Point outside image")
    size = (source.width, source.height)
    group = plan.accident_group
    mask = validate_group(group, group["members"] if group else [], run_dir, size)
    result_path = within(run_dir, f"segments/{source.source_id}/result.json")
    if result_path.exists() and read_json(result_path).get("accident_group") != group:
        raise DataError("Plan differs from the source accident selection")
    if (
        plan.segment_id != group["group_id"]
        or plan.segment_mask_path != group["mask_path"]
        or plan.segment_mask_hash != group["mask_sha256"]
    ):
        raise DataError("Plan accident group identity changed")
    actual = load_mask(run_dir, plan.region_mask_path, plan.mask_sha256, size)
    if not np.array_equal(mask, actual):
        raise DataError("Edit mask must equal the full accident group")
    ys, xs = np.nonzero(mask)
    if not np.allclose(plan.source_point, [xs.mean(), ys.mean()], rtol=0, atol=1e-8):
        raise DataError("Start must be the full group centroid")
    delta = plan.operation_params["displacement_pixels"]
    if (
        len(delta) != 2
        or any(type(v) is not int for v in delta)
        or not np.allclose(np.subtract(plan.target_point, plan.source_point), delta, rtol=0, atol=1e-8)
    ):
        raise DataError("One common integer displacement is required")
    members = [load_mask(run_dir, m["mask_path"], m["mask_sha256"], size) for m in group["members"]]
    if group_grid_geometry(mask, delta, members) != plan.operation_params["group_grid_geometry"]:
        raise DataError("Group feature geometry changed")
    if int(mask.sum()) != plan.area_pixels or not math.isclose(
        plan.area_pixels / (size[0] * size[1]), plan.range_fraction_of_image
    ):
        raise DataError("Edit area metadata mismatch")
