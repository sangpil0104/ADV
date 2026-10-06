import copy
import json
import math
from pathlib import Path

import pytest
from PIL import Image

from advv.config import load_config, validate_config
from advv.contracts import Generated, RawResponse
from advv.errors import BackendError
from advv.storage import atomic_image, load_rgb, pixel_hash, within
from advv.verifiers.clip_filter import load_sentences

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(tmp_path):
    images = tmp_path / "input"
    images.mkdir()
    Image.new("RGB", (120, 80), (40, 60, 80)).save(images / "a.png")
    config = load_config(PROJECT / "configs/advv.example.yaml", input_dir=images, target_count=1, gpus="0,1")
    config["run"]["output_root"] = str(tmp_path / "runs")
    config["report"]["terminal_progress"] = False
    return config


@pytest.fixture
def v2cfg(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["sampler"]["version"] = "object_region_v2"
    for key, name in (
        ("prompt_path", "prompts/source_profile_v4.txt"),
        ("response_schema", "schemas/source_profile_v4.schema.json"),
    ):
        cfg["source_profile"][key] = str(PROJECT / name)
    cfg["_assets"]["profile_prompt"] = (PROJECT / "prompts/source_profile_v4.txt").read_text()
    cfg["_assets"]["profile_schema"] = json.loads(
        (PROJECT / "schemas/source_profile_v4.schema.json").read_text()
    )
    validate_config(cfg)
    return cfg


SOURCE_SIMILARITY = 0.15  # Fixture cosine of every source image with every sentence.
REJECT = {0: 0.2}  # Candidate fixture: sentence 0 rises 0.2 over the source, far above max_delta.


def unit_vector(sims):
    """A unit vector whose cosine with text basis vector i is exactly sims[i]."""
    rest = 1.0 - sum(x * x for x in sims)
    assert rest > 0, "fixture similarities must fit in a unit vector"
    return [*sims, math.sqrt(rest)]


class FakeBackend:
    """Test-only fixture. Production CLI has no fake mode.

    deltas: one entry per candidate CLIP call; None (default once exhausted) keeps every sentence at the
    source similarity, {index: delta} raises sentences, "invalid" returns a wrong-dimension embedding, and an
    exception is raised. Sentence i embeds as basis vector e_i, so similarities are exact fixtures.
    """

    provenance = "fake"

    def __init__(
        self,
        deltas=None,
        colors=None,
        *,
        uncertain_source=False,
        subjects=None,
        parts=None,
        proposals=None,
        sentence_count=None,
    ):
        self.deltas = list(deltas or [])
        self.colors = list(colors or [(200, 10, 30)])
        self.uncertain_source = uncertain_source
        self.subjects = subjects
        self.parts = parts
        self.proposals = proposals  # Raw proposal-model output (numpy masks), or an exception to raise.
        self.proposal_calls = []
        self.generations = []
        self.calls = []
        self.embeds = []
        self.sentences = sentence_count or len(load_sentences(PROJECT / "prompts/implausible_physics_v1.txt"))
        self.closed = False

    def complete(self, images, prompt, max_new_tokens, *, receipt=None):
        self.calls.append(
            {"images": list(images), "hashes": [pixel_hash(p) for p in images], "prompt": prompt}
        )
        profile = {
            "summary": "A visibly cracked part.",
            "must_preserve": ["visible crack"],
            "uncertain": self.uncertain_source,
        }
        if self.subjects is not None:
            profile["subjects"] = self.subjects
            profile["parts"] = self.parts or []
        return RawResponse(json.dumps(profile), {"backend": "fake"})

    def embed(self, images, texts, *, receipt=None):
        self.embeds.append({"images": list(images), "texts": list(texts), "receipt": receipt})
        self.calls.append({"images": list(images), "texts": list(texts)})
        n = self.sentences
        if texts:
            assert len(texts) == n
            basis = [[float(i == j) for j in range(n + 1)] for i in range(n)]
            return {"text_embeddings": basis, "image_embeddings": [], "info": {"backend": "fake"}}
        vectors = []
        for path in images:
            sims = [SOURCE_SIMILARITY] * n
            if "candidates" in Path(path).parts:
                value = self.deltas.pop(0) if self.deltas else None
                if isinstance(value, BaseException):
                    raise value
                if value == "invalid":
                    return {"text_embeddings": [], "image_embeddings": [[1.0, 0.0]], "info": {}}
                for index, delta in (value or {}).items():
                    sims[index] += delta
            vectors.append(unit_vector(sims))
        return {"text_embeddings": [], "image_embeddings": vectors, "info": {"backend": "fake"}}

    def propose(self, source, subjects, parts, run_dir):
        self.proposal_calls.append((source.source_id, list(subjects), list(parts)))
        if self.proposals is None:
            raise BackendError("Fixture backend has no region proposals")
        if isinstance(self.proposals, BaseException):
            raise self.proposals
        return {
            "parts": list(parts),
            **self.proposals,
            "backend": "fixture",
            "models": {"fixture": "test-only"},
        }

    def generate(self, source, plan, run_dir):
        self.generations.append((source.source_id, plan.attempt_index, plan.seed))
        color = self.colors.pop(0)
        if isinstance(color, BaseException):
            raise color
        image = (
            load_rgb(within(run_dir, source.image_path))
            if color == "source"
            else Image.new("RGB", (source.width, source.height), color)
        )
        output = within(run_dir, f"candidates/{plan.edit_id}/generated.png")
        atomic_image(output, image)
        effective = {
            "source_point": plan.source_point,
            "target_point": plan.target_point,
            "anchor_point": plan.anchor_point,
            "region_mask_path": plan.region_mask_path,
            "backend_input_size": [source.width, source.height],
            "backend": "fake",
        }
        return Generated(output, effective, {"backend": "fake"})

    def close(self):
        self.closed = True
