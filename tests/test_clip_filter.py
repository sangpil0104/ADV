"""CLIP implausible-scene filter (T026): delta rule, frozen references, resume, legacy runs. CPU only."""

import copy
import io
import json
import signal

import pytest
import yaml

from advv import cli
from advv.config import load_config, validate_config
from advv.human_review import review_run
from advv.errors import BackendError, ConfigError, DataError, FatalBackendError, ResponseError
from advv.pipeline import Pipeline, create_run
from advv.reporting import export_run, records_for, report_run
from advv.storage import atomic_json, digest, file_hash, read_json
from advv.verifiers.clip_filter import decide, judge, load_sentences, normalized, similarities
from conftest import PROJECT, REJECT, SOURCE_SIMILARITY, FakeBackend
from test_human_review import as_production

TEXTS = ["a car flying in the sky", "a person's body stuck into the ground", "an animal floating in the air"]


def run(cfg, backend, run_id="clip"):
    root = create_run(cfg, run_id, provenance="fake")
    return root, Pipeline(root, backend).run()


def with_sentences(cfg, path):
    cfg = copy.deepcopy(cfg)
    cfg["verifier"]["sentences_path"] = str(path)
    sentences = load_sentences(path)
    cfg["_assets"].update(
        implausible_sentences=sentences,
        implausible_sentences_sha256=digest(sentences),
        implausible_sentences_file_sha256=file_hash(path),
    )
    validate_config(cfg)
    return cfg


# Exact binary fractions, so the boundary is not blurred by rounding.
@pytest.mark.parametrize(
    "candidate, verdict", [(0.3125, "accepted"), (0.3125 + 2**-20, "rejected"), (0.0, "accepted")]
)
def test_delta_equal_to_limit_passes_and_above_rejects(candidate, verdict):
    result = judge(TEXTS[:1], [0.25], [candidate], 0.0625)
    assert result["verdict"] == verdict
    assert result["sentences"][0]["delta"] == candidate - 0.25
    assert result["exceeded"] == ([0] if verdict == "rejected" else [])


def test_any_single_sentence_over_the_limit_rejects():
    result = judge(TEXTS, [0.2, 0.2, 0.2], [0.21, 0.3, 0.1], 0.05)
    assert result["verdict"] == "rejected" and result["exceeded"] == [1]
    assert result["top"] == {"index": 1, "text": TEXTS[1], "delta": pytest.approx(0.1)}
    assert judge(TEXTS, [0.2, 0.2, 0.2], [0.24, 0.25, 0.1], 0.05)["verdict"] == "accepted"


def test_judge_refuses_mismatched_or_non_finite_scores():
    with pytest.raises(DataError):
        judge(TEXTS, [0.2, 0.2], [0.2, 0.2, 0.2], 0.05)
    with pytest.raises(DataError):
        judge(TEXTS[:1], [0.2], [float("nan")], 0.05)


def test_decide_states_and_tampered_judgment():
    assert decide({"checks": {}}) == "pending"
    assert decide({"checks": {"clip": {"status": "error"}}}) == "verification_error"
    record = {
        "checks": {"clip": {"status": "completed", "response": {"similarities": [0.5]}}},
        "clip_filter": judge(TEXTS[:1], [0.2], [0.5], 0.05),
    }
    assert decide(record) == "rejected"
    record["clip_filter"]["verdict"] = "accepted"
    with pytest.raises(DataError, match="inconsistent"):
        decide(record)


def tampered(path, value):
    def apply(record):
        target = record["clip_filter"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return apply


def consistent_candidate(record):
    # Candidate scores and every derived field changed together; only the check response disagrees.
    rows = record["clip_filter"]["sentences"]
    record["clip_filter"].update(
        judge([r["text"] for r in rows], [r["source"] for r in rows], [0.0] * len(rows), 0.05)
    )


@pytest.mark.parametrize(
    "tamper, message",
    [
        (tampered(("sentences", 1, "delta"), 0.0), "inconsistent"),
        (tampered(("sentences", 1, "exceeds"), False), "inconsistent"),
        (tampered(("sentences", 0, "exceeds"), True), "inconsistent"),
        (tampered(("top",), {"index": 0, "text": TEXTS[0], "delta": 0.01}), "inconsistent"),
        (consistent_candidate, "check response"),
    ],
)
def test_decide_recomputes_every_stored_row(tamper, message):
    result = judge(TEXTS, [0.2, 0.2, 0.2], [0.21, 0.3, 0.1], 0.05)
    record = {
        "checks": {"clip": {"status": "completed", "response": {"similarities": [0.21, 0.3, 0.1]}}},
        "clip_filter": copy.deepcopy(result),
    }
    assert decide(record) == "rejected"
    tamper(record)
    with pytest.raises(DataError, match=message):
        decide(record)


def test_sentence_file_format(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("# comment\n\n  a car flying in the sky  \nwater # not a comment\n")
    assert load_sentences(path) == ["a car flying in the sky", "water # not a comment"]
    path.write_text("a\na\n")
    with pytest.raises(ConfigError, match="Duplicate"):
        load_sentences(path)
    path.write_text("# only comments\n")
    with pytest.raises(ConfigError, match="No implausible"):
        load_sentences(path)
    shipped = load_sentences(PROJECT / "prompts/implausible_physics_v1.txt")
    assert 10 <= len(shipped) <= 20 and TEXTS[0] in shipped and TEXTS[1] in shipped


def test_embedding_validation_and_cosine():
    with pytest.raises(ResponseError):
        normalized([[1.0, 0.0]], count=2)
    with pytest.raises(ResponseError):
        normalized([[0.0, 0.0]], count=1)
    with pytest.raises(ResponseError):
        normalized([[1.0, 0.0]], count=1, dim=3)
    texts = normalized([[1.0, 0.0], [0.0, 2.0]], count=2)
    image = normalized([[3.0, 4.0]], count=1)[0]
    assert similarities(image, texts) == pytest.approx([0.6, 0.8])


@pytest.mark.parametrize("offset, status", [(-0.001, "accepted"), (0.001, "rejected")])
def test_pipeline_margin_around_the_limit(cfg, offset, status):
    limit = cfg["verifier"]["max_delta"]
    root, state = run(cfg, FakeBackend([{4: limit + offset}]))
    record = records_for(root)[0]
    assert record["status"] == status
    row = record["clip_filter"]["sentences"][4]
    assert row["source"] == pytest.approx(SOURCE_SIMILARITY) and row["delta"] == pytest.approx(limit + offset)
    assert record["clip_filter"]["model_revision"] == cfg["verifier"]["model_revision"]
    assert record["clip_filter"]["sentences_sha256"] == cfg["_assets"]["implausible_sentences_sha256"]


def test_one_of_many_sentences_rejects_and_report_counts_it(cfg):
    cfg["run"]["target_count"] = 1
    backend = FakeBackend([{2: 0.1}, {2: 0.1, 5: 0.2}, {1: -0.1}], [(1, 1, 1), (2, 2, 2), (3, 3, 3)])
    root, state = run(cfg, backend)
    assert [r["status"] for r in records_for(root)] == ["rejected", "rejected", "accepted"]
    report = read_json(report_run(root).with_suffix(".json"))["clip_filter"]
    sentences = cfg["_assets"]["implausible_sentences"]
    assert report["rejected"] == 2 and report["judged"] == 3
    assert report["rejected_by_sentence"] == {sentences[2]: 2, sentences[5]: 1}
    assert report["top_sentence_of_rejected"] == {sentences[2]: 1, sentences[5]: 1}
    assert report["delta_by_sentence"][sentences[2]]["n"] == 3
    assert sentences[2] in (root / "report.md").read_text()


def test_source_reference_is_frozen_once_and_reused(cfg, tmp_path):
    from PIL import Image

    Image.new("RGB", (100, 90), (90, 30, 30)).save(tmp_path / "input/b.png")
    cfg["run"]["target_count"] = 3
    backend = FakeBackend([REJECT, None, KeyboardInterrupt()], [(1, 2, 3), (4, 5, 6), (7, 8, 9)])
    root, state = run(cfg, backend)
    assert state["status"] == "interrupted"
    # One text call and one call per source, all before the first candidate.
    assert [len(e["texts"]) > 0 for e in backend.embeds[:3]] == [True, False, False]
    references = sorted((root / "clip/sources").glob("*.json"))
    assert len(references) == 2 and all(read_json(p)["status"] == "ready" for p in references)
    frozen = {p: file_hash(p) for p in [*references, root / "clip/texts.json"]}
    second = FakeBackend(None, [(10, 11, 12), (13, 14, 15)])
    assert Pipeline(root, second).run()["status"] == "completed"
    assert all(e["texts"] == [] and "candidates" in e["images"][0].parts for e in second.embeds)
    assert all(file_hash(p) == h for p, h in frozen.items())


def test_reference_failure_stops_then_resume_retries_in_a_new_round(cfg):
    class TextsFail(FakeBackend):
        def embed(self, images, texts, *, receipt=None):
            if texts:
                raise BackendError("worker hiccup")
            return super().embed(images, texts, receipt=receipt)

    root, state = run(cfg, TextsFail())
    assert state["status"] == "failed" and "clip_reference_error" in state["last_error"]["message"]
    backend = FakeBackend()
    assert Pipeline(root, backend).run()["status"] == "completed"
    texts = read_json(root / "clip/texts.json")
    assert texts["status"] == "ready" and len(texts["failed_checks"]) == 1
    assert len(texts["failed_checks"][0]["attempts"]) == 2
    assert "round_1" in texts["checks"]["texts"]["attempts"][0]["receipt"]


def test_fatal_clip_error_is_a_verification_error_and_stops(cfg):
    root, state = run(cfg, FakeBackend([FatalBackendError("CUDA out of memory")]))
    assert state["status"] == "failed"
    record = records_for(root)[0]
    assert record["status"] == "verification_error" and record["fatal"]
    assert len(record["checks"]["clip"]["attempts"]) == 1


def test_changed_sentence_file_blocks_resume(cfg, tmp_path):
    path = tmp_path / "sentences.txt"
    path.write_text("# v1\n" + "\n".join(cfg["_assets"]["implausible_sentences"]) + "\n")
    cfg = with_sentences(cfg, path)
    root, state = run(cfg, FakeBackend([KeyboardInterrupt()]))
    assert state["status"] == "interrupted"
    path.write_text(path.read_text().replace("# v1", "# comments do not change the judgment"))
    Pipeline(root, FakeBackend())  # Comment-only edits keep the same sentence list.
    path.write_text(path.read_text() + "a train floating above the tracks\n")
    with pytest.raises(DataError, match="sentence file changed"):
        Pipeline(root, FakeBackend())
    path.unlink()
    with pytest.raises(DataError, match="unreadable"):
        Pipeline(root, FakeBackend())


def test_export_rows_carry_per_sentence_scores(cfg):
    root, _ = run(cfg, FakeBackend([{0: 0.01}]))
    as_production(root)
    (row,) = [json.loads(line) for line in export_run(root).read_text().splitlines()]
    f = row["clip_filter"]
    assert f["verdict"] == "accepted" and f["top"]["text"] == cfg["_assets"]["implausible_sentences"][0]
    assert len(f["sentences"]) == len(cfg["_assets"]["implausible_sentences"])
    assert set(f["sentences"][0]) == {"text", "source", "candidate", "delta"}
    assert f["sentences_file_sha256"] == cfg["_assets"]["implausible_sentences_file_sha256"]
    # A record whose source scores disagree with the frozen reference is not exported.
    record = records_for(root)[0]
    record["clip_filter"]["sentences"][0]["source"] -= 0.5
    record["clip_filter"]["sentences"][0]["delta"] += 0.5
    record["clip_filter"]["sentences"][0]["exceeds"] = True
    record["clip_filter"].update(verdict="rejected", exceeded=[0])
    atomic_json(root / f"records/{record['candidate_id']}.json", record)
    with pytest.raises(DataError):
        export_run(root)


def test_export_refuses_rows_off_the_frozen_references(cfg):
    root, _ = run(cfg, FakeBackend([{0: 0.01}]))
    as_production(root)
    export_run(root)
    (path,) = (root / "records").glob("*.json")
    original = read_json(path)
    # Candidate scores rewritten consistently with the verdict, but not with the stored check response.
    record = copy.deepcopy(original)
    record["clip_filter"]["sentences"][0]["candidate"] += 0.001
    record["clip_filter"]["sentences"][0]["delta"] += 0.001
    record["clip_filter"]["top"]["delta"] += 0.001
    atomic_json(path, record)
    with pytest.raises(DataError, match="check response"):
        export_run(root)
    # Row texts that are not the run's frozen sentence list.
    record = copy.deepcopy(original)
    rows = record["clip_filter"]["sentences"]
    rows[0]["text"], rows[1]["text"] = rows[1]["text"], rows[0]["text"]
    record["clip_filter"]["top"]["text"] = rows[record["clip_filter"]["top"]["index"]]["text"]
    atomic_json(path, record)
    with pytest.raises(DataError, match="frozen references"):
        export_run(root)
    # The frozen texts.json itself no longer matching the run config.
    atomic_json(path, original)
    texts = read_json(root / "clip/texts.json")
    atomic_json(root / "clip/texts.json", {**texts, "sentences": texts["sentences"][::-1]})
    with pytest.raises(DataError, match="Frozen CLIP sentence list"):
        export_run(root)


@pytest.mark.parametrize(
    "key, value",
    [("max_delta", -0.01), ("max_delta", 2), ("max_delta", "0.03"), ("similarity", "logit"), ("backend", "qwen")],
)
def test_verifier_config_validation(cfg, key, value):
    bad = copy.deepcopy(cfg)
    bad["verifier"][key] = value
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_legacy_vqa_config_is_refused_with_migration_hint(cfg, tmp_path):
    data = yaml.safe_load((PROJECT / "configs/advv.example.yaml").read_text())
    data["verifier"] = {"backend": "qwen_multimodal_local", "physical_prompt_path": "x.txt"}
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="T026"):
        load_config(path)
    bad = copy.deepcopy(cfg)
    bad["decision"] = {"require_all_yes": ["physical", "semantic"]}
    with pytest.raises(ConfigError, match="decision"):
        validate_config(bad)


def test_legacy_vqa_run_is_refused_clearly(cfg):
    root, _ = run(cfg, FakeBackend())
    config = read_json(root / "config.json")
    config["verifier"] = {"backend": "qwen_multimodal_local"}
    config.pop("profiler")
    atomic_json(root / "config.json", config)
    actions = (
        lambda: Pipeline(root, FakeBackend()),
        lambda: export_run(root),
        lambda: report_run(root),
        lambda: review_run(root, decisions=[], out=io.StringIO()),
    )
    for action in actions:
        with pytest.raises(DataError, match="T026"):
            action()


def test_cli_resume_of_a_legacy_run_reports_the_t026_error(cfg, monkeypatch, caplog):
    monkeypatch.setattr(signal, "signal", lambda *args: None)  # cli.main must not leave a SIGTERM handler.
    root, _ = run(cfg, FakeBackend())
    config = read_json(root / "config.json")
    config["verifier"] = {"backend": "qwen_multimodal_local"}
    config.pop("profiler")
    atomic_json(root / "config.json", config)

    def no_preflight(*args, **kwargs):
        raise AssertionError("preflight must not run for a pre-T026 run")

    monkeypatch.setattr(cli, "preflight", no_preflight)
    assert cli.main(["run", "--resume", "--run-dir", str(root), "--gpus", "0"]) == 1
    assert "DataError" in caplog.text and "T026" in caplog.text
    assert "KeyError" not in caplog.text
