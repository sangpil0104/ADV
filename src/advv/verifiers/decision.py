"""Paired cosine policy, independent of CUDA and the model implementation."""

from __future__ import annotations

import math

from ..errors import ResponseError
from ..storage import digest


def validate_criteria(criteria: dict) -> None:
    expected = {
        "version": "clip_criteria_v1",
        "language": "en",
        "score": "cosine_similarity",
        "pair_pass": "preferred_minus_undesired > 0",
        "physical_ids": [1, 2, 3, 4],
        "minimum_physical_passes": 3,
        "mandatory_ids": [5],
        "require_all_scores_finite": True,
    }
    if any(criteria.get(k) != v for k, v in expected.items()):
        raise ResponseError("Invalid CLIP policy: require 3 of 1-4 AND mandatory 5, margin > 0")
    pairs = criteria.get("criteria", [])
    if len(pairs) != 5 or [p.get("id") for p in pairs] != list(range(1, 6)):
        raise ResponseError("CLIP criteria must contain ordered IDs 1..5")
    texts = [p.get(k) for p in pairs for k in ("undesired", "preferred")]
    if any(not isinstance(t, str) or not t.strip() for t in texts) or len(set(texts)) != 10:
        raise ResponseError("CLIP requires ten distinct nonempty captions")


def evaluate_scores(scores, *, criteria_sha256: str = "", info=None) -> dict:
    if not isinstance(scores, (list, tuple)) or len(scores) != 10:
        raise ResponseError("CLIP must return all ten cosine scores")
    if any(type(s) not in (int, float) or not math.isfinite(s) or abs(s) > 1.00001 for s in scores):
        raise ResponseError("CLIP scores must be finite raw cosine values")
    pairs = []
    for i in range(5):
        a, b = float(scores[2 * i]), float(scores[2 * i + 1])
        pairs.append(
            {"id": i + 1, "undesired_score": a, "preferred_score": b, "margin": b - a, "passed": b > a}
        )
    physical = sum(p["passed"] for p in pairs[:4])
    return {
        "pairs": pairs,
        "physical_pass_count": physical,
        "accident_passed": pairs[4]["passed"],
        "passed": physical >= 3 and pairs[4]["passed"],
        "criteria_sha256": criteria_sha256,
        "info": info or {},
    }


def decide(result: dict) -> str:
    try:
        scores = [p[k] for p in result["pairs"] for k in ("undesired_score", "preferred_score")]
        if [p["id"] for p in result["pairs"]] != list(range(1, 6)):
            raise ResponseError("Incorrect CLIP criterion order")
        rebuilt = evaluate_scores(
            scores, criteria_sha256=result.get("criteria_sha256", ""), info=result.get("info", {})
        )
        if digest(rebuilt) != digest(result):
            raise ResponseError("Stored CLIP decision or margins disagree with scores")
    except (KeyError, TypeError, ValueError) as exc:
        raise ResponseError("Invalid CLIP result") from exc
    return "accepted" if rebuilt["passed"] else "rejected"
