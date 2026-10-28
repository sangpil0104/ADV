"""Pinned offline SegFormer Cityscapes ground segmentation, in the ADVV environment."""

from pathlib import Path

import numpy as np

from ..errors import ConfigError
from ..ground import GROUND_LABELS, save_ground
from ..storage import file_hash


class GroundSegmenter:
    def __init__(self, cfg, root, *, device="cuda:0"):
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.torch, self.settings, self.root, self.device = torch, cfg["ground"], root, device
        settings = self.settings
        folder = Path(settings["model_path"])
        for name, checksum in settings["files_sha256"].items():
            if file_hash(folder / name) != checksum:
                raise ConfigError("Ground model asset changed")
        self.processor = SegformerImageProcessor.from_pretrained(folder, local_files_only=True)
        self.model = SegformerForSemanticSegmentation.from_pretrained(folder, local_files_only=True)
        if any(self.model.config.id2label[i] != GROUND_LABELS[i] for i in settings["class_ids"]):
            raise ConfigError("Ground model class mapping changed")
        self.model.eval().to(device)

    def __call__(self, image, source_id):
        torch = self.torch
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            logits = torch.nn.functional.interpolate(
                logits, size=(image.height, image.width), mode="bilinear", align_corners=False
            )
            confidence, labels = logits.softmax(dim=1).max(dim=1)
            selected = torch.zeros_like(labels, dtype=torch.bool)
            for class_id in self.settings["class_ids"]:
                selected |= labels == class_id
            selected &= confidence >= self.settings["confidence_threshold"]
            mask = selected[0].cpu().numpy().astype(np.bool_)
        return save_ground(
            self.root,
            source_id,
            mask,
            self.settings,
            info={
                "backend": self.settings["backend"],
                "input_policy": "pinned_processor_bilinear_logits_to_original_align_corners_false",
                "confidence_threshold": self.settings["confidence_threshold"],
            },
        )
