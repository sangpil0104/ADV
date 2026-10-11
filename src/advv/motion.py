"""CPU appearance matching diagnostics; never an extra candidate acceptance gate."""

import numpy as np
from PIL import Image

from .segmentation import load_mask


def diagnose_motion(original, generated, group, delta, root, max_side=512):
    import cv2

    factor = min(1.0, max_side / max(original.size))
    size = tuple(max(1, round(v * factor)) for v in original.size)
    scale = [size[i] / original.size[i] for i in (0, 1)]
    a = np.asarray(original.convert("L").resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
    b = np.asarray(generated.convert("L").resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
    results = []
    for member in group["members"]:
        mask = load_mask(root, member["mask_path"], member["mask_sha256"], original.size)
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(size, Image.Resampling.NEAREST)) > 0
        ys, xs = np.nonzero(mask)
        row = {
            "segment_id": member["segment_id"],
            "status": "unmeasurable",
            "expected_displacement_pixels": delta,
        }
        if len(xs) >= 4:
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            support = mask[y0:y1, x0:x1].astype(np.uint8)
            template = a[y0:y1, x0:x1]
            if float(template[support > 0].std()) > 1e-6:
                try:
                    scores = cv2.matchTemplate(b, template, cv2.TM_CCORR_NORMED, mask=support)
                except cv2.error:
                    row["status"] = "matching_error"
                    results.append(row)
                    continue
                finite = np.isfinite(scores)
                if finite.any():
                    yy, xx = np.unravel_index(np.argmax(np.where(finite, scores, -np.inf)), scores.shape)
                    shift = [float(xx - x0) / scale[0], float(yy - y0) / scale[1]]
                    row.update(
                        status="appearance_match_only",
                        estimated_displacement_pixels=shift,
                        normalized_correlation=float(scores[yy, xx]),
                        residual_pixels=float(np.linalg.norm(np.subtract(shift, delta))),
                    )
        results.append(row)
    return {
        "method": "masked_template_match_v1",
        "status": "diagnostic_only_not_motion_verification",
        "affects_acceptance": False,
        "matching_size": list(size),
        "members": results,
        "limitations": "Appearance matches can confuse similar vehicles, occlusions, or regenerated texture; no movement/preservation threshold is defined.",
    }
