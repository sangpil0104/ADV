"""CLIP implausible-scene filter: pure CPU math over stored embeddings, no model import."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..errors import ConfigError, DataError, ResponseError
from ..storage import digest, read_json, safe_id, within

SIMILARITY = "cosine"  # Of L2-normalized projected CLIP embeddings; CLIP's logit_scale is not applied.
CRITERION = "max_delta_over_source"
ACCEPTED, REJECTED = "accepted", "rejected"


def load_sentences(path: Path) -> list[str]:
    """One sentence per line; blank lines and lines starting with '#' are ignored, nothing else is."""
    sentences = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text in sentences:
            raise ConfigError(f"Duplicate implausible sentence in {path}: {text}")
        sentences.append(text)
    if not sentences:
        raise ConfigError(f"No implausible sentences in {path}")
    return sentences


def normalized(vectors, *, count: int, dim: int | None = None) -> np.ndarray:
    """Model output as unit rows; a wrong shape or a non-finite/zero vector is a technical error."""
    try:
        array = np.asarray(vectors, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ResponseError(f"Invalid CLIP embedding output: {exc}") from exc
    if array.ndim != 2 or array.shape[0] != count or array.shape[1] < 1:
        raise ResponseError(f"Expected {count} CLIP embeddings, got shape {array.shape}")
    if dim is not None and array.shape[1] != dim:
        raise ResponseError(f"CLIP embedding dimension {array.shape[1]} != text dimension {dim}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.all(np.isfinite(array)) or not np.all(norms > 0):
        raise ResponseError("CLIP embedding is non-finite or zero")
    return array / norms


def similarities(image_embedding: np.ndarray, text_embeddings: np.ndarray) -> list[float]:
    return [float(x) for x in text_embeddings @ image_embedding]


def judge(sentences: list[str], source: list[float], candidate: list[float], limit: float) -> dict:
    """Reject when any sentence's similarity rose over the source by more than the limit (equal passes)."""
    if not (len(sentences) == len(source) == len(candidate)):
        raise DataError("CLIP similarity lists do not match the sentence list")
    if not all(math.isfinite(x) for x in [*source, *candidate, limit]):
        raise DataError("Non-finite CLIP similarity or limit")
    rows = []
    for index, (text, s, c) in enumerate(zip(sentences, source, candidate)):
        delta = c - s
        rows.append(
            {"index": index, "text": text, "source": s, "candidate": c, "delta": delta, "exceeds": delta > limit}
        )
    top = max(rows, key=lambda r: (r["delta"], -r["index"]))
    exceeded = [r["index"] for r in rows if r["exceeds"]]
    return {
        "verdict": REJECTED if exceeded else ACCEPTED,
        "limit": limit,
        "exceeded": exceeded,
        "top": {"index": top["index"], "text": top["text"], "delta": top["delta"]},
        "sentences": rows,
    }


def decide(record: dict) -> str:
    """accepted / rejected only from a completed check whose stored judgment recomputes identically."""
    check = record.get("checks", {}).get("clip", {})
    if check.get("status") == "error":
        return "verification_error"
    result = record.get("clip_filter")
    if check.get("status") != "completed" or not result:
        return "pending"
    rows = result["sentences"]
    if [r["candidate"] for r in rows] != check.get("response", {}).get("similarities"):
        raise DataError(f"Stored CLIP candidate scores differ from the check response: {record.get('candidate_id')}")
    again = judge(
        [r["text"] for r in rows], [r["source"] for r in rows], [r["candidate"] for r in rows], result["limit"]
    )
    # Every derived field (per-sentence delta/exceeds, top, verdict) must recompute exactly.
    if any(result.get(k) != v for k, v in again.items()):
        raise DataError(f"Stored CLIP filter judgment is inconsistent: {record.get('candidate_id')}")
    return result["verdict"]


def frozen_sentences(run_dir: Path, sentences: list[str], sentences_sha256: str) -> None:
    """The run's frozen clip/texts.json must hold exactly the configured sentence list."""
    texts = read_json(within(run_dir, "clip/texts.json", must_exist=True))
    if texts["sentences"] != sentences or digest(texts["sentences"]) != sentences_sha256:
        raise DataError("Frozen CLIP sentence list differs from the run config")


def frozen_source(run_dir: Path, source_id: str) -> list[float]:
    ref = read_json(within(run_dir, f"clip/sources/{safe_id(source_id)}.json", must_exist=True))
    return ref["checks"]["source"]["response"]["similarities"]


def check_frozen(record: dict, sentences: list[str], source: list[float]) -> None:
    """A stored judgment is valid only over the run's frozen sentences and its source's frozen scores."""
    rows = record["clip_filter"]["sentences"]
    if [r["text"] for r in rows] != sentences or [r["source"] for r in rows] != source:
        raise DataError(f"CLIP filter rows differ from the frozen references: {record.get('candidate_id')}")


def summary(record: dict) -> dict | None:
    """Short form for previews and export rows."""
    result = record.get("clip_filter")
    if not result:
        return None
    return {
        "verdict": result["verdict"],
        "limit": result["limit"],
        "top": result["top"],
        "exceeded": [result["sentences"][i]["text"] for i in result["exceeded"]],
    }
