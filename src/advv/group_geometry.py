"""Whole-mask translation geometry, independent of model imports and CUDA."""

from __future__ import annotations

import numpy as np

from .errors import DataError


def feature_grid(width, height):
    full_w, full_h = width // 16 * 16, height // 16 * 16
    scale = 4 if (full_w + full_h) / 2 > 1000 else 3 if (full_w + full_h) / 2 > 600 else 2
    return full_w // scale, full_h // scale


def resize_region(mask, size):
    """Bilinear align_corners=False, the pinned Dragger's region interpolation."""
    w, h = size
    if min(w, h) <= 0:
        raise DataError("Image is too small for a DragFlow feature grid")
    yy = np.clip((np.arange(h, dtype=np.float32) + 0.5) * (mask.shape[0] / h) - 0.5, 0, mask.shape[0] - 1)
    xx = np.clip((np.arange(w, dtype=np.float32) + 0.5) * (mask.shape[1] / w) - 0.5, 0, mask.shape[1] - 1)
    y0, x0 = yy.astype(int), xx.astype(int)
    y1, x1 = np.minimum(y0 + 1, mask.shape[0] - 1), np.minimum(x0 + 1, mask.shape[1] - 1)
    fy, fx = (yy - y0)[:, None], (xx - x0)[None, :]
    values = mask.astype(np.float32)
    return (
        values[y0[:, None], x0] * (1 - fy) * (1 - fx)
        + values[y1[:, None], x0] * fy * (1 - fx)
        + values[y0[:, None], x1] * (1 - fy) * fx
        + values[y1[:, None], x1] * fy * fx
    ).astype(np.float32)


def translation_fits(mask, delta):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return False
    dx, dy = delta
    return (
        0 <= xs.min() + dx
        and xs.max() + dx < mask.shape[1]
        and 0 <= ys.min() + dy
        and ys.max() + dy < mask.shape[0]
    )


def group_grid_geometry(mask, delta, members=()):
    h, w = mask.shape
    grid = feature_grid(w, h)
    region = resize_region(mask, grid)
    ys, xs = np.nonzero(region > 0.5)
    if not len(xs):
        raise DataError("Group vanished at the DragFlow feature resolution")
    for member in members:
        if not (resize_region(member, grid) > 0.5).any():
            raise DataError("A selected vehicle vanished at the DragFlow feature resolution")
    fitted_delta = [round(delta[i] * grid[i] / (w, h)[i]) for i in (0, 1)]
    if (
        fitted_delta == [0, 0]
        or not translation_fits(mask, delta)
        or not translation_fits(region > 0, fitted_delta)
    ):
        raise DataError("Whole group translation is empty or leaves the image/feature grid")
    start = [round(float(xs.mean())), round(float(ys.mean()))]
    return {
        "grid_size": list(grid),
        "grid_start": start,
        "grid_delta": fitted_delta,
        "grid_end": [start[i] + fitted_delta[i] for i in (0, 1)],
    }


def translate_mask(mask, dx, dy):
    """Integer translation without wraparound; only repair support may clip at image edges."""
    h, w = mask.shape
    result = np.zeros_like(mask)
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    if sx0 < sx1 and sy0 < sy1:
        result[sy0 + dy : sy1 + dy, sx0 + dx : sx1 + dx] = mask[sy0:sy1, sx0:sx1]
    return result
