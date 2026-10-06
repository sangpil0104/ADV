from __future__ import annotations

from pathlib import Path

from ..errors import BackendError
from ..storage import digest, load_rgb


class Clip:
    """Unit-norm projected CLIP embeddings; similarity and the filter decision stay in the coordinator."""

    def __init__(self, cfg, device: str = "cuda:0"):
        import torch
        import transformers
        from transformers import CLIPModel, CLIPProcessor

        self.torch, self.cfg, self.device = torch, cfg["verifier"], device
        self.version = transformers.__version__
        self.processor = CLIPProcessor.from_pretrained(self.cfg["model_path"], local_files_only=True)
        self.model = (
            CLIPModel.from_pretrained(
                self.cfg["model_path"], dtype=getattr(torch, self.cfg["dtype"]), local_files_only=True
            )
            .to(device)
            .eval()
        )
        self.max_tokens = self.model.config.text_config.max_position_embeddings
        self.processor_hash = digest(self.processor.image_processor.to_dict())

    def _unit(self, output):
        # Transformers 5 returns the projected embedding as pooler_output; 4.x returns the tensor itself.
        features = getattr(output, "pooler_output", output).float()
        if not bool(self.torch.isfinite(features).all()):
            raise BackendError("CLIP produced a non-finite embedding")
        return features / features.norm(dim=-1, keepdim=True)

    def __call__(self, request):
        torch = self.torch
        texts, images = request["texts"], [load_rgb(Path(p)) for p in request["images"]]
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        result = {"text_embeddings": [], "image_embeddings": []}
        with torch.inference_mode():
            if texts:
                tokens = self.processor.tokenizer(texts, padding=True, return_tensors="pt")
                if tokens["input_ids"].shape[-1] > self.max_tokens:
                    # Silent truncation would judge a different sentence than the one on record.
                    raise BackendError(f"A sentence exceeds CLIP's {self.max_tokens}-token limit")
                result["text_embeddings"] = self._unit(self.model.get_text_features(**tokens.to(self.device))).tolist()
            if images:
                pixels = self.processor(images=images, return_tensors="pt")["pixel_values"]
                pixels = pixels.to(self.device, dtype=self.model.dtype)
                result["image_embeddings"] = self._unit(self.model.get_image_features(pixel_values=pixels)).tolist()
        result["info"] = {
            "backend": "clip_local",
            "model_id": self.cfg["model_id"],
            "model_revision": self.cfg["model_revision"],
            "dtype": self.cfg["dtype"],
            "transformers": self.version,
            "torch": torch.__version__,
            "image_processor_sha256": self.processor_hash,
            "image_sizes": [list(im.size) for im in images],
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if self.device.startswith("cuda") else 0,
        }
        return result
