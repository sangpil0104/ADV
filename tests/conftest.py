from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from advv.config import load_config
from advv.accident_group import build_group, selection_identity
from advv.contracts import Generated
from advv.storage import atomic_image, digest, file_hash, load_rgb, pixel_hash, within
from advv.verifiers.decision import evaluate_scores

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(tmp_path):
    folder = tmp_path / "input"
    folder.mkdir()
    Image.new("RGB", (120, 80), (40, 60, 80)).save(folder / "a.png")
    config = load_config(PROJECT / "configs/advv.example.yaml", input_dir=folder, target_count=1, gpus="0,1")
    config["run"]["output_root"] = str(tmp_path / "runs")
    config["report"]["terminal_progress"] = False
    return config


def fixture_segments(source, root, cfg, *, empty=False):
    segments = []
    if not empty:
        mask = Image.new("L", (source.width, source.height))
        w, h = source.width, source.height
        ImageDraw.Draw(mask).polygon(
            [
                (w * 0.15, h * 0.3),
                (w * 0.4, h * 0.15),
                (w * 0.7, h * 0.22),
                (w * 0.84, h * 0.62),
                (w * 0.7, h * 0.8),
                (w * 0.2, h * 0.68),
            ],
            fill=255,
        )
        path = f"segments/{source.source_id}/fixture.png"
        atomic_image(within(root, path), mask)
        segments = [
            {
                "segment_id": source.source_id + "_car",
                "mask_path": path,
                "mask_sha256": file_hash(within(root, path)),
                "class_id": 3,
            }
        ]
    selection = {
        "identity": selection_identity(cfg),
        "decisions": [{"segment_id": e["segment_id"], "selected": True} for e in segments],
        "method": "test_fixture",
    }
    return {
        "segments": segments,
        "selection": selection,
        "accident_group": build_group(
            source.source_id, (source.width, source.height), segments, selection, root
        ),
        "source_sha256": source.pixel_sha256,
        "settings_sha256": digest(cfg["segmentation"]),
        "info": {"backend": "fake"},
    }


class FakeBackend:
    """Test-only, never eligible for production export."""

    provenance = "fake"

    def __init__(self, cfg, outcomes=None, colors=None, *, empty_segments=False, timer=None, durations=None):
        self.cfg = cfg
        self.outcomes = list(outcomes) if outcomes is not None else None
        self.colors = list(colors) if colors is not None else None
        self.empty_segments = empty_segments
        self.timer, self.durations = timer, list(durations or [])
        self.generations, self.calls, self.segment_calls = [], [], []
        self.closed = False

    def segment(self, source, run_dir):
        self.segment_calls.append(source.source_id)
        return fixture_segments(source, run_dir, self.cfg, empty=self.empty_segments)

    def generate(self, source, plan, run_dir):
        self.generations.append(plan.to_dict())
        if self.durations:
            self.timer.advance(self.durations.pop(0))
        color = self.colors.pop(0) if self.colors is not None else (len(self.generations), 15, 37)
        if isinstance(color, BaseException):
            raise color
        image = (
            load_rgb(within(run_dir, source.image_path))
            if color == "source"
            else Image.new("RGB", (source.width, source.height), color)
        )
        path = within(run_dir, f".staging/{plan.edit_id}/backend/generated.png")
        atomic_image(path, image)
        effective = {
            "source_point": plan.source_point,
            "target_point": plan.target_point,
            "anchor_point": None,
            "region_mask_path": plan.region_mask_path,
            "mask_sha256": plan.mask_sha256,
            "backend_input_size": [source.width, source.height],
        }
        return Generated(path, effective, {"backend": "fake"})

    def verify(self, image, *, receipt):
        self.calls.append({"image": str(image), "pixel_sha256": pixel_hash(image)})
        outcome = self.outcomes.pop(0) if self.outcomes is not None else True
        if isinstance(outcome, BaseException):
            raise outcome
        bits = outcome if isinstance(outcome, list) else [True, True, True, True, outcome]
        scores = [score for bit in bits for score in (0.1, 0.3 if bit else 0.0)]
        return evaluate_scores(
            scores, criteria_sha256=self.cfg["_assets"]["criteria_sha256"], info={"backend": "fake"}
        )

    def close(self):
        self.closed = True


class ManualClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds
