"""Pinned torchvision COCO Mask R-CNN; outputs instance masks, not boxes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..accident_group import build_group, select_vehicles
from ..storage import atomic_image, digest, file_hash, load_rgb, within


class VehicleSegmenter:
    def __init__(self, cfg, run_dir, *, device="cuda:0"):
        import torch
        from torchvision.models.detection import MaskRCNN
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        from torchvision.ops.misc import FrozenBatchNorm2d

        self.torch, self.device, self.cfg, self.root = torch, device, cfg, run_dir
        # Build offline with the same FrozenBatchNorm backbone as the COCO_V1 factory.
        backbone = resnet_fpn_backbone(
            backbone_name="resnet50", weights=None, norm_layer=FrozenBatchNorm2d, trainable_layers=0
        )
        self.model = MaskRCNN(backbone, num_classes=91, min_size=800, max_size=1333)
        weights = Path(cfg["segmentation"]["weights_path"])
        if file_hash(weights) != cfg["segmentation"]["weights_sha256"]:
            raise ValueError("Segmentation checkpoint hash changed")
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        for module in self.model.modules():
            if isinstance(module, FrozenBatchNorm2d):
                module.eps = 0.0  # COCO_V1's official overwrite_eps setting.
        self.model.eval().to(device)

    def __call__(self, request):
        from torchvision.transforms.functional import to_tensor

        source = request["source"]
        image = load_rgb(within(self.root, source["image_path"], must_exist=True))
        self.model.to(self.device)
        with self.torch.inference_mode():
            result = self.model([to_tensor(image).to(self.device)])[0]
        settings = self.cfg["segmentation"]
        segments = []
        for i, (label, score, mask) in enumerate(zip(result["labels"], result["scores"], result["masks"])):
            label, score = int(label), float(score)
            if label not in settings["class_ids"] or score < settings["score_threshold"]:
                continue
            arr = mask[0].detach().cpu().numpy() >= settings["mask_threshold"]
            if int(arr.sum()) < settings["min_area_pixels"]:
                continue
            sid = f"{source['source_id']}_segment_{i:04d}"
            relative = f"segments/{source['source_id']}/{sid}.png"
            atomic_image(within(self.root, relative), Image.fromarray(arr.astype(np.uint8) * 255))
            segments.append(
                {
                    "segment_id": sid,
                    "mask_path": relative,
                    "mask_sha256": file_hash(within(self.root, relative)),
                    "class_id": label,
                    "confidence": score,
                    "area_pixels": int(arr.sum()),
                }
            )
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        from .clip import CLIPVerifier
        from ..visualization import render_selection

        captions = [
            self.cfg["_assets"]["accident_selection"]["criteria"][kind][key]
            for kind in ("single", "contact_group")
            for key in ("uninvolved", "involved")
        ]
        scorer = CLIPVerifier(self.cfg, device=self.device, captions=captions) if segments else None
        selection = select_vehicles(image, segments, self.root, self.cfg, scorer)
        group = build_group(source["source_id"], image.size, segments, selection, self.root)
        # Release the large vehicle/CLIP models before the separate semantic pass.
        self.model.to("cpu")
        del scorer
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
        from .ground import GroundSegmenter

        ground = GroundSegmenter(self.cfg, self.root, device=self.device)(image, source["source_id"])
        from ..visualization import render_ground

        ground["preview_path"] = render_ground(self.root, source, ground)
        preview = render_selection(self.root, source, segments, selection)
        return {
            "segments": segments,
            "selection": selection,
            "accident_group": group,
            "ground": ground,
            "selection_preview": preview,
            "source_sha256": source["pixel_sha256"],
            "settings_sha256": digest(settings),
            "info": {
                "backend": settings["backend"],
                "weights_sha256": settings["weights_sha256"],
                "input_policy": "rgb_0_1_min800_max1333",
            },
        }
