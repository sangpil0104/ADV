"""Source-level accident vehicle selection; separate from candidate CLIP verification."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .errors import DataError
from .segmentation import load_mask
from .storage import atomic_image, digest, file_hash, within


SELECTION_LIMITATIONS = (
    "Undetected or partially detected vehicles (e.g. overturned cars) cannot be selected; "
    "projected contact is not physical contact; rule thresholds were set from 10 development images."
)
FILTER_LIMITATIONS = (
    " Detection filters (v3) can also drop a real accident vehicle that is low-score, fragmented by "
    "occlusion, very large, or reaching the top edge; such a source then moves an incomplete group."
)
RULE_SELECTORS = ("clip_contact_context_v2", "clip_contact_context_v3")


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


def contact_components(masks, gap_fraction=None):
    """Projected overlap/touch is an association cue, never proof of an accident.

    With ``gap_fraction`` two masks also count as touching when their pixel gap is at most
    that fraction of the smaller mask's bbox diagonal (never below 8-neighbor touch).
    Masks never grow.
    """
    from scipy import ndimage

    neighbors = {i: {i} for i in range(len(masks))}
    if gap_fraction is not None:
        diagonals = [math.hypot(x1 - x0, y1 - y0) for x0, y0, x1, y1 in map(_box, masks)]
    for i, mask in enumerate(masks):
        if gap_fraction is None:
            touching = ndimage.binary_dilation(mask, structure=np.ones((3, 3)))
        else:
            distance = ndimage.distance_transform_edt(~mask)
        for j in range(i + 1, len(masks)):
            if gap_fraction is None:
                joined = np.any(touching & masks[j])
            else:
                tolerance = max(1.5, gap_fraction * min(diagonals[i], diagonals[j]))
                joined = distance[masks[j]].min() <= tolerance
            if joined:
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


def _box(mask):
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def ego_hood(mask, size, rules):
    """Camera car's own hood/dashboard: wide, large and reaching the bottom edge."""
    width, height = size
    x0, _, x1, y1 = _box(mask)
    return (
        y1 >= height * (1 - rules["ego_bottom_margin_fraction"])
        and x1 - x0 >= width * rules["ego_min_width_fraction"]
        and mask.sum() >= width * height * rules["ego_min_area_fraction"]
    )


def vehicle_units(masks, segments, candidates, min_iou):
    """Merge high-IoU detections of the same vehicle; the vehicle keeps their union."""
    parent = {i: i for i in candidates}

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for a, i in enumerate(candidates):
        for j in candidates[a + 1 :]:
            union = np.logical_or(masks[i], masks[j]).sum()
            if np.logical_and(masks[i], masks[j]).sum() >= min_iou * union:
                parent[root(j)] = root(i)
    units = {}
    for i in candidates:
        units.setdefault(root(i), []).append(i)

    def rank(i):
        return (-float(segments[i].get("confidence", 0.0)), -int(masks[i].sum()), i)

    return [sorted(members, key=rank) for members in units.values()]


def distant_from(vmasks, v, anchor, rules):
    """Much smaller and standing much farther up the image (= farther away) than the anchor."""
    _, y0, _, bottom = _box(vmasks[anchor])
    area = int(vmasks[anchor].sum())
    gap = bottom - _box(vmasks[v])[3]
    if vmasks[v].sum() < rules["distant_max_area_ratio"] * area and gap > (
        rules["distant_min_bottom_gap_fraction"] * (bottom - y0)
    ):
        return {
            "anchor_vehicle": anchor,
            "area_ratio": float(vmasks[v].sum() / area),
            "bottom_gap_pixels": gap,
            "anchor_bbox_height": bottom - y0,
            "overlap_fraction_of_own_area": float(
                np.logical_and(vmasks[v], vmasks[anchor]).sum() / vmasks[v].sum()
            ),
        }
    return None


def detection_filter(mask, confidence, size, rules):
    """Reasons (in rule order) a merged vehicle unit looks like a false detection, plus its metrics."""
    from scipy import ndimage

    width, height = size
    _, y0, _, y1 = _box(mask)
    area = int(mask.sum())
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3)))
    metrics = {
        "confidence": confidence,
        "area_fraction": area / (width * height),
        "bbox_top_fraction": y0 / height,
        "bbox_bottom_fraction": y1 / height,
        "largest_component_fraction": int(np.bincount(labels.ravel())[1:].max()) / area,
    }
    checks = (
        ("low_detection_score", confidence < rules["detection_min_score"]),
        ("mask_too_large", metrics["area_fraction"] > rules["max_area_fraction"]),
        ("mask_too_small", metrics["area_fraction"] < rules["min_area_fraction"]),
        (
            "elevated_top_structure",
            y0 < rules["top_touch_margin_fraction"] * height
            and y1 <= rules["top_touch_max_bottom_fraction"] * height,
        ),
        ("above_ground_level", y1 <= rules["min_bottom_fraction"] * height),
        (
            "fragmented_mask",
            metrics["largest_component_fraction"] < rules["min_largest_component_fraction"],
        ),
    )
    return [name for name, hit in checks if hit], metrics


def _largest(vmasks, members):
    return max(members, key=lambda v: (int(vmasks[v].sum()), -v))


def detach_distant(vehicles, vmasks, gap_fraction, rules):
    """Small far vehicles overlap a crash car only through perspective; split them off."""
    active, detached = list(range(len(vehicles))), {}
    changed = True
    while changed:
        changed = False
        for component in contact_components([vmasks[v] for v in active], gap_fraction):
            members = [active[k] for k in component]
            anchor = _largest(vmasks, members)
            for v in members:
                info = None if v == anchor else distant_from(vmasks, v, anchor, rules)
                if info:
                    detached[v], changed = info, True
        active = [v for v in active if v not in detached]
    return active, detached


def _score(image, mask, single, cfg, scorer):
    views = vehicle_views(image, mask, cfg["accident_selection"]["context_padding"])
    scores = {}
    caption_index = 0 if single else 2
    for name, view in views.items():
        values = scorer.score_image(view)
        if len(values) != 4 or any(
            type(v) not in (float, int) or not math.isfinite(v) or abs(v) > 1 for v in values
        ):
            raise DataError("Invalid accident selection CLIP scores")
        a, b = values[caption_index : caption_index + 2]
        scores[name] = {"uninvolved": a, "involved": b, "margin": b - a}
    return scores


def select_vehicles(image, segments, root, cfg, scorer):
    masks = [load_mask(root, e["mask_path"], e["mask_sha256"], image.size) for e in segments]
    if cfg["accident_selection"]["method"] == "clip_contact_context_v1":
        return _select_v1(image, segments, masks, cfg, scorer)
    return _select_rules(image, segments, masks, cfg, scorer)


def _select_v1(image, segments, masks, cfg, scorer):
    components = contact_components(masks)
    decisions, associations = {}, []
    for component in components:
        mask = np.logical_or.reduce([masks[i] for i in component])
        scores = _score(image, mask, len(component) == 1, cfg, scorer)
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


def _select_rules(image, segments, masks, cfg, scorer):
    method = cfg["accident_selection"]["method"]
    if method not in RULE_SELECTORS:
        raise DataError(f"Unsupported accident selector {method}")
    rules = cfg["accident_selection"]["rules"]
    if method == "clip_contact_context_v3" and any(
        type(e.get("confidence")) not in (int, float) or not math.isfinite(e["confidence"]) for e in segments
    ):
        raise DataError("Detection filter requires a finite detection confidence per segment")
    ids = [e["segment_id"] for e in segments]
    decisions = {
        i: {"segment_id": ids[i], "selected": False, "association_index": None} for i in range(len(ids))
    }
    candidates = []
    for i, mask in enumerate(masks):
        if ego_hood(mask, image.size, rules):
            decisions[i]["excluded_reason"] = "ego_vehicle_hood"
        else:
            candidates.append(i)
    vehicles = vehicle_units(masks, segments, candidates, rules["duplicate_min_iou"])
    for members in vehicles:
        for i in members[1:]:
            decisions[i]["duplicate_of"] = ids[members[0]]
    if method == "clip_contact_context_v3":
        vehicles = _filter_detections(vehicles, masks, segments, decisions, image.size, rules)
    vmasks = [np.logical_or.reduce([masks[i] for i in v]) for v in vehicles]
    gap_fraction = rules["contact_gap_fraction"]
    active, detached = detach_distant(vehicles, vmasks, gap_fraction, rules)

    def mark(v, reason, info):
        for i in vehicles[v]:
            decisions[i].update(selected=False, excluded_reason=reason)
            decisions[i]["distant"] = {
                **{k: x for k, x in info.items() if k != "anchor_vehicle"},
                "anchor_segment_id": ids[vehicles[info["anchor_vehicle"]][0]],
            }

    for v, info in detached.items():
        mark(v, "distant_perspective_overlap", info)
    associations, units = [], []
    for component in contact_components([vmasks[v] for v in active], gap_fraction):
        unit = [active[k] for k in component]
        mask = np.logical_or.reduce([vmasks[v] for v in unit])
        scores = _score(image, mask, len(unit) == 1, cfg, scorer)
        passed = all(v["margin"] > 0 for v in scores.values())
        associations.append(
            {
                "member_ids": [ids[i] for v in unit for i in vehicles[v]],
                "vehicles": [[ids[i] for i in vehicles[v]] for v in unit],
                "scores": scores,
                "caption_pair": "single" if len(unit) == 1 else "contact_group",
                "clip_passed": passed,
                "selected": passed,
            }
        )
        units.append(unit)
        for v in unit:
            for i in vehicles[v]:
                decisions[i].update(selected=passed, association_index=len(associations) - 1)
                if not passed:
                    decisions[i]["excluded_reason"] = "clip_not_involved"
    chosen = [k for k, a in enumerate(associations) if a["selected"]]
    if chosen:
        # Separate CLIP-passing units that are tiny and far behind the core crash are background traffic.
        anchor = _largest(vmasks, [v for k in chosen for v in units[k]])
        for k in chosen:
            if anchor not in units[k] and all(distant_from(vmasks, v, anchor, rules) for v in units[k]):
                associations[k].update(selected=False, excluded_reason="distant_separate_unit")
                for v in units[k]:
                    mark(v, "distant_separate_unit", distant_from(vmasks, v, anchor, rules))
    return {
        "identity": selection_identity(cfg),
        "method": method,
        "decisions": [decisions[i] for i in range(len(ids))],
        "associations": associations,
        "association_policy": "overlap_touch_or_relative_gap_no_mask_growth",
        "rules": dict(rules),
        "status": "proxy_selection_requires_review",
        "limitations": SELECTION_LIMITATIONS
        + (FILTER_LIMITATIONS if method == "clip_contact_context_v3" else ""),
        "candidate_acceptance_rule_changed": False,
    }


def _filter_detections(vehicles, masks, segments, decisions, size, rules):
    """Drop whole vehicle units (duplicates merged) before contact analysis; never split a unit."""
    kept = []
    for members in vehicles:
        mask = np.logical_or.reduce([masks[i] for i in members])
        reasons, metrics = detection_filter(mask, float(segments[members[0]]["confidence"]), size, rules)
        for i in members:
            decisions[i]["detection_filter"] = {"reasons": reasons, **metrics}
            if reasons:
                decisions[i]["excluded_reason"] = reasons[0]
        if not reasons:
            kept.append(members)
    return kept


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
