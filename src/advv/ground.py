"""Ground segmentation records and deterministic, per-vehicle placement constraints.

Contact points are a 2-D lower-boundary proxy, not measured wheel contacts or 3-D physics.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .errors import DataError
from .group_geometry import feature_grid, resize_region
from .segmentation import load_mask
from .storage import atomic_image, digest, file_hash, within

GROUND_MODEL = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
GROUND_REVISION = "21b3847fae21ddee674abd31129307b6a1235bd9"
GROUND_LABELS = {0: "road", 1: "sidewalk", 9: "terrain"}
PLACEMENT_VERSION = "lower_boundary_ground_v1"


def save_ground(root, source_id, mask, settings, *, info=None):
    relative = f"segments/{source_id}/ground.png"
    atomic_image(within(root, relative), Image.fromarray(mask.astype(np.uint8) * 255))
    return {
        "mask_path": relative,
        "mask_sha256": file_hash(within(root, relative)),
        "area_pixels": int(mask.sum()),
        "status": "ready" if mask.any() else "empty",
        "settings_sha256": digest(settings),
        "model_id": settings["model_id"],
        "model_revision": settings["model_revision"],
        "class_ids": settings["class_ids"],
        "classes": [GROUND_LABELS[i] for i in settings["class_ids"]],
        "info": info or {},
    }


def validate_ground(root, ground, size, settings=None):
    if not isinstance(ground, dict):
        raise DataError("Missing ground segmentation; no unrestricted fallback")
    mask = load_mask(root, ground["mask_path"], ground["mask_sha256"], size)
    if ground["area_pixels"] != int(mask.sum()) or ground["status"] != ("ready" if mask.any() else "empty"):
        raise DataError("Ground area/status changed")
    if settings is not None and ground["settings_sha256"] != digest(settings):
        raise DataError("Ground segmentation settings changed")
    return mask


def contact_points(mask, count, band_fraction):
    """Sample the lower silhouette; each point is one pixel BELOW the vehicle mask."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise DataError("Empty vehicle has no contact proxy")
    cutoff = ys.max() - max(1, round((ys.max() - ys.min() + 1) * band_fraction)) + 1
    columns = np.unique(xs[ys >= cutoff])
    selected = np.unique(np.rint(np.linspace(0, len(columns) - 1, count)).astype(int))
    return [[int(columns[i]), int(np.flatnonzero(mask[:, columns[i]])[-1] + 1)] for i in selected]


def group_contacts(group, root, size, settings):
    return [
        {
            "segment_id": member["segment_id"],
            "points": contact_points(
                load_mask(root, member["mask_path"], member["mask_sha256"], size),
                settings["contact_points_per_vehicle"],
                settings["contact_band_fraction"],
            ),
        }
        for member in group["members"]
    ]


def _ground_at(ground, xx, yy):
    """Conservative subpixel membership: all neighboring floor/ceil pixels must be ground."""
    result = np.ones((len(yy), len(xx)), dtype=bool)
    h, w = ground.shape
    for rounding in (np.floor, np.ceil):
        x = rounding(xx).astype(int)
        for y_rounding in (np.floor, np.ceil):
            y = y_rounding(yy).astype(int)
            result &= (
                (y[:, None] >= 0)
                & (y[:, None] < h)
                & (x[None, :] >= 0)
                & (x[None, :] < w)
                & ground[np.clip(y, 0, h - 1)[:, None], np.clip(x, 0, w - 1)[None, :]]
            )
    return result


def feasible_translations(mask, members, ground, contacts):
    """Enumerate all integer common shifts, including the upstream feature rounding."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    dx = np.arange(-int(xs.min()), w - int(xs.max()))
    dy = np.arange(-int(ys.min()), h - int(ys.max()))
    gw, gh = feature_grid(w, h)
    region = resize_region(mask, (gw, gh))
    if not (region > 0.5).any() or any(not (resize_region(m, (gw, gh)) > 0.5).any() for m in members):
        raise DataError("A selected vehicle vanished at the DragFlow feature resolution")
    gy, gx = np.nonzero(region > 0)
    grid_dx, grid_dy = np.rint(dx * gw / w).astype(int), np.rint(dy * gh / h).astype(int)
    feasible = ((grid_dx[None, :] != 0) | (grid_dy[:, None] != 0)) & (
        (gx.min() + grid_dx[None, :] >= 0)
        & (gx.max() + grid_dx[None, :] < gw)
        & (gy.min() + grid_dy[:, None] >= 0)
        & (gy.max() + grid_dy[:, None] < gh)
    )
    for member in contacts:
        for x, y in member["points"]:
            feasible &= _ground_at(ground, x + dx, y + dy)
            feasible &= _ground_at(ground, x + grid_dx * w / gw, y + grid_dy * h / gh)
    return feasible, [int(dx[0]), int(dy[0])]


def prepare_placement(source, result, cfg, root):
    settings, group = cfg["ground"], result.get("accident_group")
    size = (source.width, source.height)
    ground = validate_ground(root, result.get("ground"), size, settings)
    if not group:
        return {"status": "no_accident_group"}
    contacts = group_contacts(group, root, size, settings)
    identity = digest([PLACEMENT_VERSION, size, group, result["ground"], settings, contacts])
    cached = result.get("ground_placement")
    if cached:
        if cached.get("identity") != identity or cached.get("contacts") != contacts:
            raise DataError("Ground placement cache changed")
        feasible = load_mask(root, cached["mask_path"], cached["mask_sha256"], cached["size"])
        if int(feasible.sum()) != cached["count"] or cached["status"] != (
            "ready" if feasible.any() else "no_valid_ground_translation"
        ):
            raise DataError("Ground placement count/status changed")
        return cached
    mask = load_mask(root, group["mask_path"], group["mask_sha256"], size)
    members = [load_mask(root, m["mask_path"], m["mask_sha256"], size) for m in group["members"]]
    feasible, offset = feasible_translations(mask, members, ground, contacts)
    relative = f"segments/{source.source_id}/ground_translations.png"
    atomic_image(within(root, relative), Image.fromarray(feasible.astype(np.uint8) * 255))
    return {
        "version": PLACEMENT_VERSION,
        "identity": identity,
        "status": "ready" if feasible.any() else "no_valid_ground_translation",
        "count": int(feasible.sum()),
        "offset_xy": offset,
        "size": [feasible.shape[1], feasible.shape[0]],
        "mask_path": relative,
        "mask_sha256": file_hash(within(root, relative)),
        "contacts": contacts,
    }


def placement_evidence(ground_record, contacts, settings, delta, grid_geometry, size, root):
    ground = validate_ground(root, ground_record, size, settings)
    effective = [grid_geometry["grid_delta"][i] * size[i] / grid_geometry["grid_size"][i] for i in (0, 1)]
    evidence = []
    for member in contacts:
        targets = {}
        for name, shift in (("requested", delta), ("effective", effective)):
            points = [[x + shift[0], y + shift[1]] for x, y in member["points"]]
            if any(not _ground_at(ground, np.array([x]), np.array([y])).all() for x, y in points):
                raise DataError("A vehicle contact proxy leaves segmented ground")
            targets[name] = points
        evidence.append(member | {"targets": targets})
    return {
        "method": PLACEMENT_VERSION,
        "ground": ground_record,
        "settings": settings,
        "contact_points_per_vehicle": settings["contact_points_per_vehicle"],
        "contact_band_fraction": settings["contact_band_fraction"],
        "contacts": evidence,
        "effective_delta": effective,
        "status": "planned_contacts_on_ground",
        "scope": "placement_constraint_only_not_generated_contact_verification",
    }
