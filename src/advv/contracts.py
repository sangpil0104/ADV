from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Source:
    source_id: str
    image_path: str
    original_path: str
    file_sha256: str
    pixel_sha256: str
    width: int
    height: int
    split: str = "source"
    group_id: str = ""
    domain: str = "dashcam_accident"
    preserve_hint: str | None = None
    label: dict | None = None
    group_inferred: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EditRequest:
    source_id: str
    edit_id: str
    attempt_index: int
    seed: int
    sampler_version: str
    operation: str
    operation_params: dict
    region_mask_path: str
    source_point: list[float]
    target_point: list[float]
    anchor_point: list[float] | None
    source_prompt: str
    target_prompt: str
    mask_sha256: str
    segment_id: str = ""
    segment_mask_path: str = ""
    segment_mask_hash: str = ""
    range_fraction_of_image: float = 0.0
    area_pixels: int = 0
    augmentation_prompt: str | None = None
    accident_group: dict | None = None
    schema_version: str = "2.0"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Generated:
    path: Path
    effective: dict
    info: dict = field(default_factory=dict)


class Backend(Protocol):
    provenance: str

    def segment(self, source: Source, run_dir: Path) -> dict: ...
    def verify(self, image: Path, *, receipt: str) -> dict: ...
    def generate(self, source: Source, plan: EditRequest, run_dir: Path) -> Generated: ...
    def close(self) -> None: ...
