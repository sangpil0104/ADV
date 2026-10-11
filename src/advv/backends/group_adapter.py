"""Feed full, possibly disconnected masks to the pinned official Dragger.

The dashboard contour loader is deliberately bypassed, without modifying upstream.
The per-worker coordinate hook keeps its centroid replacement from changing delta.
"""

from contextlib import contextmanager

import numpy as np
from scipy import ndimage

from ..errors import BackendError
from ..group_geometry import group_grid_geometry, translate_mask


def repair_support(mask, delta, grid_delta, grid_size):
    h, w = mask.shape
    effective_delta = [round(grid_delta[i] * (w, h)[i] / grid_size[i]) for i in (0, 1)]
    support = mask.copy()
    for dx, dy in (delta, effective_delta):
        steps = max(1, int(np.ceil(max(abs(dx), abs(dy)) / 8)))
        for i in range(1, steps + 1):
            support |= translate_mask(mask, round(dx * i / steps), round(dy * i / steps))
    # This support is only for background restoration, never the moving object mask.
    return ndimage.binary_dilation(support, iterations=8)


def attach_group_masks(instruction, mask, plan, *, device, dtype):
    import torch

    geometry = group_grid_geometry(mask, plan.operation_params["displacement_pixels"])
    op = instruction["region_operations"]["0"]
    op["region"] = torch.as_tensor(mask.astype(np.float32), device=device, dtype=dtype)[None, None]
    op["raw_weight"] = op["norm_weight"] = torch.tensor(1.0, device=device, dtype=dtype)
    support = repair_support(
        mask, plan.operation_params["displacement_pixels"], geometry["grid_delta"], geometry["grid_size"]
    )
    instruction["mask"] = torch.as_tensor(support.astype(np.float32), device=device, dtype=dtype)[None, None]
    instruction["source_mask"] = None
    instruction["advv_group_geometry"] = geometry
    return support


@contextmanager
def group_coordinate_hook(dragger_module):
    original = dragger_module.scale_coordinates

    def scale(instruction, raw_shape, cut_shape, update_forward=True):
        import torch
        from torch.nn import functional as F

        result = original(instruction, raw_shape, cut_shape, update_forward=update_forward)
        expected = result.get("advv_group_geometry")
        if not expected or not update_forward:
            return result
        if list(reversed(cut_shape)) != expected["grid_size"]:
            raise BackendError("Unexpected DragFlow feature grid for the accident group")
        op = result["region_operations"]["0"]
        region = F.interpolate(op["region"], size=cut_shape, mode="bilinear", align_corners=False)
        ys, xs = torch.where(region[0, 0] > 0.5)
        if not len(xs):
            raise BackendError("Empty accident group at backend feature resolution")
        start = torch.stack((xs.float().mean(), ys.float().mean())).round().long()
        if start.tolist() != expected["grid_start"]:
            raise BackendError("Accident group centroid differs from planned feature geometry")
        delta = torch.tensor(expected["grid_delta"], device=start.device)
        op["points_fit"] = [start, start.clone(), start + delta]
        op["full_delta"] = tuple(expected["grid_delta"])
        return result

    dragger_module.scale_coordinates = scale
    try:
        yield
    finally:
        dragger_module.scale_coordinates = original
