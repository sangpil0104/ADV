"""Local CLIP embedding similarity; no generated text and no automatic downloads."""

from __future__ import annotations

from PIL import Image, ImageOps

from ..errors import ResponseError
from ..storage import load_rgb
from ..verifiers.decision import evaluate_scores, validate_criteria


def letterbox(image, side):
    scaled = ImageOps.contain(image.convert("RGB"), (side, side), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (side, side), (127, 127, 127))
    canvas.paste(scaled, ((side - scaled.width) // 2, (side - scaled.height) // 2))
    return canvas


class CLIPVerifier:
    def __init__(self, cfg, *, device="cuda:0", captions=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch, self.device, self.cfg = torch, device, cfg
        settings = cfg["verifier"]
        self.model = (
            CLIPModel.from_pretrained(settings["model_path"], local_files_only=True).eval().to(device)
        )
        self.processor = CLIPProcessor.from_pretrained(settings["model_path"], local_files_only=True)
        criteria = cfg["_assets"]["clip_criteria"]
        validate_criteria(criteria)
        captions = captions or [p[k] for p in criteria["criteria"] for k in ("undesired", "preferred")]
        tokens = self.processor.tokenizer(captions, padding=True, truncation=False, return_tensors="pt")
        if tokens["input_ids"].shape[1] > self.model.config.text_config.max_position_embeddings:
            raise ResponseError("CLIP caption exceeds the text context; refusing truncation")
        with torch.inference_mode():
            text = self.model.get_text_features(**tokens.to(device)).float()
            self.text_features = self._normalize(text)
        self.side = self.model.config.vision_config.image_size

    def _normalize(self, features):
        torch = self.torch
        norms = features.norm(dim=-1, keepdim=True)
        if not torch.isfinite(features).all() or not torch.isfinite(norms).all() or (norms <= 0).any():
            raise ResponseError("Invalid CLIP embedding")
        return features / norms

    def score_image(self, image):
        torch = self.torch
        frame = letterbox(image, self.side)
        inputs = self.processor.image_processor(
            images=frame, do_resize=False, do_center_crop=False, return_tensors="pt"
        )
        with torch.inference_mode():
            features = self._normalize(self.model.get_image_features(**inputs.to(self.device)).float())
            scores = (features @ self.text_features.T)[0].cpu().tolist()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return scores

    def __call__(self, request):
        from pathlib import Path

        scores = self.score_image(load_rgb(Path(request["image"])))
        return evaluate_scores(
            scores,
            criteria_sha256=self.cfg["_assets"]["criteria_sha256"],
            info={
                "model_id": self.cfg["verifier"]["model_id"],
                "model_revision": self.cfg["verifier"]["model_revision"],
                "preprocessing": self.cfg["verifier"]["preprocessing"],
                "input_size": [self.side, self.side],
                "padding_rgb": [127, 127, 127],
            },
        )
