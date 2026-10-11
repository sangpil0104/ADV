"""Source-level accident vehicle selection; separate from candidate CLIP verification."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .errors import DataError
from .segmentation import load_mask
from .storage import atomic_image, digest, file_hash, within


def selection_identity(cfg):
    return digest(
        {
            "settings": cfg["accident_selection"],
            "captions": cfg["_assets"]["accident_selection_sha256"],
            "model": {k: cfg["verifier"][k] for k in ("model_id", "model_revision", "preprocessing")},
        }
    )


def vehicle_views(image, mask, context_padding):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise DataError("Cannot classify an empty vehicle mask")
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    dx, dy = math.ceil((x1 - x0) * context_padding), math.ceil((y1 - y0) * context_padding)
    box = (max(0, x0 - dx), max(0, y0 - dy), min(image.width, x1 + dx), min(image.height, y1 + dy))
    foreground = np.full((image.height, image.width, 3), 127, dtype=np.uint8)
    foreground[mask] = np.asarray(image)[mask]
    return {"foreground": Image.fromarray(foreground).crop(box), "context": image.crop(box)}


def contact_components(masks):
    """Projected overlap/touch is an association cue, never proof of an accident."""
    from scipy import ndimage

    neighbors = {i: {i} for i in range(len(masks))}
    for i, mask in enumerate(masks):
        touching = ndimage.binary_dilation(mask, structure=np.ones((3, 3)))
        for j in range(i + 1, len(masks)):
            if np.any(touching & masks[j]):
                neighbors[i].add(j)
                neighbors[j].add(i)
    components, visited = [], set()
    for i in neighbors:
        if i in visited:
            continue
        pending, group = [i], set()
        while pending:
            j = pending.pop()
            if j in group:
                continue
            group.add(j)
            pending.extend(neighbors[j] - group)
        visited |= group
        components.append(sorted(group))
    return components


def select_vehicles(image, segments, root, cfg, scorer):
    masks = [load_mask(root, e["mask_path"], e["mask_sha256"], image.size) for e in segments]
    components = contact_components(masks)
    decisions, associations = {}, []
    for component in components:
        mask = np.logical_or.reduce([masks[i] for i in component])
        views = vehicle_views(image, mask, cfg["accident_selection"]["context_padding"])
        scores = {}
        caption_index = 0 if len(component) == 1 else 2
        for name, view in views.items():
            values = scorer.score_image(view)
            if len(values) != 4 or any(
                type(v) not in (float, int) or not math.isfinite(v) or abs(v) > 1 for v in values
            ):
                raise DataError("Invalid accident selection CLIP scores")
            a, b = values[caption_index : caption_index + 2]
            scores[name] = {"uninvolved": a, "involved": b, "margin": b - a}
        chosen = all(v["margin"] > 0 for v in scores.values())
        association = {
            "member_ids": [segments[i]["segment_id"] for i in component],
            "scores": scores,
            "caption_pair": "single" if len(component) == 1 else "contact_group",
            "selected": chosen,
        }
        associations.append(association)
        for i in component:
            decisions[i] = {
                "segment_id": segments[i]["segment_id"],
                "selected": chosen,
                "association_index": len(associations) - 1,
            }
    return {
        "identity": selection_identity(cfg),
        "method": "clip_contact_context_v1",
        "decisions": [decisions[i] for i in range(len(segments))],
        "associations": associations,
        "association_policy": "overlap_or_8_neighbor_touch_no_mask_growth",
        "status": "proxy_selection_requires_review",
        "candidate_acceptance_rule_changed": False,
    }


def build_group(source_id, size, segments, selection, root):
    ids = [d["segment_id"] for d in selection["decisions"] if d["selected"]]
    if not ids:
        return None
    members = [e for e in segments if e["segment_id"] in ids]
    if len(members) != len(ids) or len(set(ids)) != len(ids):
        raise DataError("Unknown or duplicate selected vehicle")
    union = np.zeros((size[1], size[0]), dtype=bool)
    for member in members:
        union |= load_mask(root, member["mask_path"], member["mask_sha256"], size)
    path = f"segments/{source_id}/accident_group.png"
    atomic_image(within(root, path), Image.fromarray(union.astype(np.uint8) * 255))
    return {
        "group_id": source_id + "_accident",
        "member_ids": ids,
        "members": members,
        "mask_path": path,
        "mask_sha256": file_hash(within(root, path)),
        "area_pixels": int(union.sum()),
        "selection_sha256": digest(selection),
    }


def validate_group(group, segments, root, size):
    if not group or not group.get("members"):
        raise DataError("No accident vehicle group")
    ids = [e["segment_id"] for e in group["members"]]
    if ids != group["member_ids"] or len(set(ids)) != len(ids):
        raise DataError("Invalid accident group members")
    available = {e["segment_id"]: e for e in segments}
    union = np.zeros((size[1], size[0]), dtype=bool)
    for entry in group["members"]:
        if available.get(entry["segment_id"]) != entry:
            raise DataError("Accident group member changed")
        union |= load_mask(root, entry["mask_path"], entry["mask_sha256"], size)
    actual = load_mask(root, group["mask_path"], group["mask_sha256"], size)
    if not union.any() or not np.array_equal(actual, union) or int(union.sum()) != group["area_pixels"]:
        raise DataError("Accident group must contain every full selected vehicle mask")
    return union
