from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .accident_group import validate_group
from .contracts import EditRequest, Source
from .errors import DataError
from .group_geometry import feature_grid, group_grid_geometry, resize_region
from .segmentation import load_mask
from .storage import atomic_image, digest, file_hash, read_json, within

SAMPLER_VERSION = "accident_group_v3"
MAGNITUDE_DISTRIBUTIONS = ("uniform_radius", "uniform_area")
SAMPLING_KEYS = {
    "magnitude_distribution",
    "radius_pixels",
    "angle_radians",
    "vertical_scale",
    "min_radius_pixels",
    "max_radius_pixels",
    "min_displacement_fraction",
    "min_displacement_image_diag_fraction",
    "max_displacement_fraction",
    "group_bbox_diagonal_pixels",
    "image_diagonal_pixels",
    "displacement_norm_pixels",
    "feasible_angle_radians",
}
MAX_IMAGE_FLOOR = 0.5
SETTING_KEYS = (
    "magnitude_distribution",
    "vertical_scale",
    "min_displacement_fraction",
    "min_displacement_image_diag_fraction",
    "max_displacement_fraction",
)


@dataclass(frozen=True)
class MotionSpace:
    """Bounding boxes (inclusive) and limits that decide whether an integer group shift is valid.

    ``low`` floors the rounded |delta| (after vertical_scale), ``high`` caps it; low > high means
    no valid move (the source is held, the cap is never raised to the floor).
    """

    size: tuple[int, int]
    grid: tuple[int, int]
    bbox: tuple[int, int, int, int]
    grid_bbox: tuple[int, int, int, int]
    low: float
    high: float
    scale: float


def grid_point(point, width, height):
    grid = feature_grid(width, height)
    return [round(point[i] / (width, height)[i] * grid[i]) for i in (0, 1)]


def group_diagonal(mask) -> float:
    ys, xs = np.nonzero(mask)
    return math.hypot(int(xs.max()) - int(xs.min()) + 1, int(ys.max()) - int(ys.min()) + 1)


def sample_radius(rng, low: float, high: float, distribution: str) -> float:
    u = float(rng.random())
    if distribution == "uniform_area":
        return math.sqrt(low**2 + u * (high**2 - low**2))
    return low + u * (high - low)


def polar_displacement(radius: float, angle: float, vertical_scale: float) -> list[int]:
    return [round(radius * math.cos(angle)), round(vertical_scale * radius * math.sin(angle))]


def displacement_floor(settings: dict, diagonal: float, size) -> float:
    return max(
        settings["min_displacement_fraction"] * diagonal,
        settings["min_displacement_image_diag_fraction"] * math.hypot(*size),
    )


def motion_space(mask, members, settings: dict) -> MotionSpace:
    h, w = mask.shape
    grid = feature_grid(w, h)
    region = resize_region(mask, grid)
    if not (region > 0.5).any():
        raise DataError("Group vanished at the DragFlow feature resolution")
    if any(not (resize_region(m, grid) > 0.5).any() for m in members):
        raise DataError("A selected vehicle vanished at the DragFlow feature resolution")
    ys, xs = np.nonzero(mask)
    gys, gxs = np.nonzero(region > 0)
    diagonal = group_diagonal(mask)
    return MotionSpace(
        size=(w, h),
        grid=grid,
        bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
        grid_bbox=(int(gxs.min()), int(gys.min()), int(gxs.max()), int(gys.max())),
        low=displacement_floor(settings, diagonal, (w, h)),
        high=settings["max_displacement_fraction"] * diagonal,
        scale=settings["vertical_scale"],
    )


def valid_deltas(space: MotionSpace, dx, dy):
    """Vectorized group_grid_geometry fit/nonidentity test plus the rounded-norm floor and cap."""
    (w, h), (gw, gh) = space.size, space.grid
    x0, y0, x1, y1 = space.bbox
    gx0, gy0, gx1, gy1 = space.grid_bbox
    fx, fy = np.round(dx * gw / w), np.round(dy * gh / h)
    norm = np.hypot(dx, dy)
    return (
        (norm <= space.high)
        & (norm >= space.low)
        & (x0 + dx >= 0)
        & (x1 + dx < w)
        & (y0 + dy >= 0)
        & (y1 + dy < h)
        & ~((fx == 0) & (fy == 0))
        & (gx0 + fx >= 0)
        & (gx1 + fx < gw)
        & (gy0 + fy >= 0)
        & (gy1 + fy < gh)
    )


def angle_intervals(space: MotionSpace, radius: float):
    """Disjoint [start, end) angle intervals at this radius whose rounded shift is valid."""
    if radius <= 0:
        return np.empty(0), np.empty(0)
    half = np.arange(-math.floor(radius) - 1, math.floor(radius) + 1) + 0.5
    cx = half[np.abs(half) < radius] / radius
    cy = half[np.abs(half) < space.scale * radius] / (space.scale * radius)
    a, b = np.arccos(cx), np.arcsin(cy)
    cuts = np.unique(
        np.concatenate([[0.0, 2 * math.pi], a, 2 * math.pi - a, np.mod(b, 2 * math.pi), math.pi - b])
    )
    starts, ends = cuts[:-1], cuts[1:]
    mids = (starts + ends) / 2
    dx = np.round(radius * np.cos(mids))
    dy = np.round(space.scale * radius * np.sin(mids))
    ok = (ends > starts) & valid_deltas(space, dx, dy)
    return starts[ok], ends[ok]


def pick_angle(starts, ends, u: float) -> float:
    lengths = ends - starts
    cumulative = np.cumsum(lengths)
    target = u * float(cumulative[-1])
    i = min(int(np.searchsorted(cumulative, target, side="right")), len(starts) - 1)
    return float(starts[i] + target - (cumulative[i] - lengths[i]))


def motion_feasible(space: MotionSpace) -> bool:
    """Exact: some valid integer shift has a rounding cell meeting the open radius band (low, high)."""
    if space.low > space.high:
        return False
    (w, h), (x0, y0, x1, y1) = space.size, space.bbox
    reach_x, reach_y = math.floor(space.high), math.floor(space.scale * space.high + 0.5)
    xs = np.arange(max(-x0, -reach_x), min(w - 1 - x1, reach_x) + 1)
    ys = np.arange(max(-y0, -reach_y), min(h - 1 - y1, reach_y) + 1)
    if not len(xs) or not len(ys):
        return False
    dx, dy = np.meshgrid(xs, ys)
    near_x, near_y = np.maximum(np.abs(dx) - 0.5, 0), np.maximum(np.abs(dy) - 0.5, 0)
    nearest = np.hypot(near_x, near_y / space.scale)
    farthest = np.hypot(np.abs(dx) + 0.5, (np.abs(dy) + 0.5) / space.scale)
    reachable = (nearest < space.high) & (farthest > space.low)
    return bool((reachable & valid_deltas(space, dx, dy)).any())


def source_motion_feasible(source: Source, segments: dict, cfg: dict, run_dir: Path) -> bool:
    size = (source.width, source.height)
    group = segments.get("accident_group")
    mask = validate_group(group, segments["segments"], run_dir, size)
    members = [load_mask(run_dir, m["mask_path"], m["mask_sha256"], size) for m in group["members"]]
    try:
        space = motion_space(mask, members, cfg["sampler"])
    except DataError:
        return False
    return motion_feasible(space)


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
    diagonal = group_diagonal(mask)
    space = motion_space(mask, members, settings)
    low, high, scale = space.low, space.high, space.scale
    distribution = settings["magnitude_distribution"]
    if low > high:
        raise DataError(
            f"No valid nonidentity whole accident group translation: displacement floor {low:.1f} px "
            f"exceeds the cap {high:.1f} px for this group; no partial-mask fallback"
        )
    for _ in range(settings["geometry_attempts"]):
        radius = sample_radius(rng, low, high, distribution)
        # Uniform over the directions that keep the group in frame/grid at this radius.
        starts, ends = angle_intervals(space, radius)
        if not len(starts):
            continue
        feasible = float((ends - starts).sum())
        angle = pick_angle(starts, ends, float(rng.random()))
        delta = polar_displacement(radius, angle, scale)
        if not 0 <= angle < 2 * math.pi or not low <= math.hypot(*delta) <= high:  # Rounded |delta|.
            continue
        try:
            geometry = group_grid_geometry(mask, delta, members)
        except DataError:
            continue
        break
    else:
        raise DataError(
            "No valid nonidentity whole accident group translation within the displacement cap; "
            "no partial-mask fallback"
        )
    sampling = {
        "magnitude_distribution": distribution,
        "radius_pixels": radius,
        "angle_radians": angle,
        "vertical_scale": scale,
        "min_radius_pixels": low,
        "max_radius_pixels": high,
        "min_displacement_fraction": settings["min_displacement_fraction"],
        "min_displacement_image_diag_fraction": settings["min_displacement_image_diag_fraction"],
        "max_displacement_fraction": settings["max_displacement_fraction"],
        "group_bbox_diagonal_pixels": diagonal,
        "image_diagonal_pixels": math.hypot(w, h),
        "displacement_norm_pixels": math.hypot(*delta),
        "feasible_angle_radians": feasible,
    }
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
        operation_params={
            "displacement_pixels": delta,
            "group_grid_geometry": geometry,
            "displacement_sampling": sampling,
        },
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
    validate_plan(plan, source, run_dir, settings)
    return plan


def validate_plan(plan: EditRequest, source: Source, run_dir: Path, settings: dict | None = None) -> None:
    if (
        plan.schema_version != "2.0"
        or plan.source_id != source.source_id
        or plan.operation != "relocation"
        or plan.sampler_version != SAMPLER_VERSION
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
    validate_sampling(plan.operation_params, mask, members, delta, settings)
    if int(mask.sum()) != plan.area_pixels or not math.isclose(
        plan.area_pixels / (size[0] * size[1]), plan.range_fraction_of_image
    ):
        raise DataError("Edit area metadata mismatch")


def validate_sampling(params: dict, mask, members, delta, settings: dict | None = None) -> None:
    sampling = params.get("displacement_sampling")
    if set(params) != {"displacement_pixels", "group_grid_geometry", "displacement_sampling"} or (
        not isinstance(sampling, dict) or set(sampling) != SAMPLING_KEYS
    ):
        raise DataError("Plan displacement sampling record is missing or malformed")
    numbers = [v for k, v in sampling.items() if k != "magnitude_distribution"]
    if not all(type(v) in (int, float) and math.isfinite(v) for v in numbers):
        raise DataError("Invalid displacement sampling values")
    if (
        not 0 <= sampling["min_displacement_fraction"] < sampling["max_displacement_fraction"] <= 1
        or not 0 <= sampling["min_displacement_image_diag_fraction"] <= MAX_IMAGE_FLOOR
        or not 0 < sampling["vertical_scale"] <= 1
        or sampling["magnitude_distribution"] not in MAGNITUDE_DISTRIBUTIONS
    ):
        raise DataError("Recorded displacement fractions, vertical_scale or distribution out of range")
    if settings is not None and any(sampling[k] != settings[k] for k in SETTING_KEYS):
        raise DataError("Recorded displacement sampling settings differ from the run sampler config")
    diagonal = group_diagonal(mask)
    space = motion_space(mask, members, sampling)
    low, high = space.low, space.high
    starts, ends = angle_intervals(space, sampling["radius_pixels"])

    def same(a, b):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)

    if (
        not same(sampling["group_bbox_diagonal_pixels"], diagonal)
        or not same(sampling["image_diagonal_pixels"], math.hypot(*mask.shape))
        or not same(sampling["min_radius_pixels"], low)
        or not same(sampling["max_radius_pixels"], high)
        or not 0 <= sampling["angle_radians"] < 2 * math.pi
        or not same(sampling["feasible_angle_radians"], float((ends - starts).sum()))
        or not any(a <= sampling["angle_radians"] <= b for a, b in zip(starts, ends))
        or not low <= sampling["radius_pixels"] <= high
        or polar_displacement(
            sampling["radius_pixels"], sampling["angle_radians"], sampling["vertical_scale"]
        )
        != list(delta)
        or not same(sampling["displacement_norm_pixels"], math.hypot(*delta))
        or not low <= math.hypot(*delta) <= high
    ):
        raise DataError(
            "Displacement is outside the recorded cap/floor or does not match its sampling record"
        )
