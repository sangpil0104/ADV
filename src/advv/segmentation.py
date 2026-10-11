"""Geometry utilities and validation; no model/CUDA imports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .errors import DataError
from .storage import file_hash, within


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if not count:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def load_mask(root: Path, relative: str, sha256: str, size) -> np.ndarray:
    path = within(root, relative, must_exist=True)
    if file_hash(path) != sha256:
        raise DataError("Segment/edit mask changed")
    with Image.open(path) as image:
        arr = np.asarray(image)
        if image.mode != "L" or image.size != tuple(size) or not set(np.unique(arr)) <= {0, 255}:
            raise DataError("Expected binary mask at normalized original resolution")
        return arr > 0


def validate_segments(source, result: dict, root: Path) -> None:
    seen = set()
    for entry in result.get("segments", []):
        if entry["segment_id"] in seen:
            raise DataError("Duplicate segment ID")
        seen.add(entry["segment_id"])
        arr = load_mask(root, entry["mask_path"], entry["mask_sha256"], (source.width, source.height))
        if not arr.any():
            raise DataError("Expected a nonempty full instance segment")
        if entry["class_id"] not in (3, 4, 6, 8):
            raise DataError("Expected a vehicle segment")
    if result.get("accident_group"):
        from .accident_group import validate_group
        from .storage import digest

        group, selection = result["accident_group"], result["selection"]
        if group["selection_sha256"] != digest(selection) or group["member_ids"] != [
            d["segment_id"] for d in selection["decisions"] if d["selected"]
        ]:
            raise DataError("Accident selection changed")
        validate_group(group, result["segments"], root, (source.width, source.height))
