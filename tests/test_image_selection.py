"""Single-image selection must constrain snapshots and both GPU candidate lanes."""

from pathlib import Path
import signal

import pytest
import yaml
from PIL import Image

from advv import cli
from advv.config import load_config, recipe_hash
from advv.errors import ConfigError, DataError
from advv.ingest import scan_sources
from advv.storage import file_hash, read_json
from conftest import FakeBackend


def config_file(cfg, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_single_image_parallel_total_ten_and_resume_uses_saved_selection(cfg, tmp_path, monkeypatch):
    Image.new("RGB", (120, 80), "red").save(tmp_path / "input/b.png")
    (tmp_path / "input/not_an_image.png").write_bytes(b"must never be opened")
    cfg["dataset"]["image"] = "b.png"  # CLI must override YAML selection.
    path = config_file(cfg, tmp_path)
    before = {p.name: file_hash(p) for p in (tmp_path / "input").iterdir()}
    instances = []

    class Backend(FakeBackend):
        def __init__(self, resolved, root):
            super().__init__(resolved)
            instances.append(self)

        def generate(self, source, plan, root):
            self.colors = [(plan.attempt_index + 1, 30, 50)]
            return super().generate(source, plan, root)

    monkeypatch.setattr(cli, "LocalBackend", Backend)
    monkeypatch.setattr(cli, "preflight", lambda *args, **kwargs: {"status": "ready"})
    original_handler = signal.getsignal(signal.SIGTERM)
    try:
        assert (
            cli.main(
                [
                    "run",
                    "10",
                    "--config",
                    str(path),
                    "--gpus",
                    "4,5,6,7",
                    "--image",
                    "a.png",
                    "--run-id",
                    "single",
                ]
            )
            == 0
        )
        root = tmp_path / "runs/single"
        sources = read_json(root / "sources.json")
        assert len(sources) == 1 and Path(sources[0]["original_path"]).name == "a.png"
        assert read_json(root / "config.json")["dataset"]["image"] == "a.png"
        assert read_json(root / "state.json")["accepted_count"] == 10
        assert sum(len(b.generations) for b in instances) == 10
        assert {p["source_id"] for b in instances for p in b.generations} == {sources[0]["source_id"]}
        assert before == {p.name: file_hash(p) for p in (tmp_path / "input").iterdir()}
        (tmp_path / "input/a.png").unlink()  # Resume reads the frozen source snapshot.
        assert cli.main(["run", "--resume", "--run-dir", str(root), "--gpus", "4,5,6,7"]) == 0
        assert all(not b.generations for b in instances[2:])
    finally:
        signal.signal(signal.SIGTERM, original_handler)


def test_selection_is_input_relative_independent_of_cwd_and_part_of_recipe(cfg, tmp_path, monkeypatch):
    Image.new("RGB", (120, 80), "red").save(tmp_path / "input/b.png")
    path = config_file(cfg, tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    whole = load_config(path)
    selected = load_config(path, image="a.png")
    assert len(scan_sources(whole)[0]) == 2
    assert len(scan_sources(selected)[0]) == 1
    assert recipe_hash(whole) != recipe_hash(selected)
    all_a = next(s for s in scan_sources(whole)[0] if s.original_path.endswith("a.png"))
    assert scan_sources(selected)[0][0].source_id == all_a.source_id


@pytest.mark.parametrize("selection", ["", " ", "../outside.png", "/tmp/outside.png"])
def test_invalid_selection(cfg, tmp_path, selection):
    with pytest.raises(ConfigError, match="--image"):
        load_config(config_file(cfg, tmp_path), image=selection)


@pytest.mark.parametrize("selection", ["missing.png", "notes.txt", "folder", "escape.png"])
def test_missing_unsupported_directory_or_escaped_symlink_never_falls_back(cfg, tmp_path, selection):
    (tmp_path / "input/notes.txt").write_text("not an image")
    (tmp_path / "input/folder").mkdir()
    Image.new("RGB", (120, 80)).save(tmp_path / "outside.png")
    (tmp_path / "input/escape.png").symlink_to(tmp_path / "outside.png")
    with pytest.raises(DataError):
        load_config(config_file(cfg, tmp_path), image=selection)


def test_cannot_bypass_manifest_with_image_selection(cfg, tmp_path):
    cfg["dataset"].update(input_dir=None, manifest=str(tmp_path / "sources.jsonl"))
    with pytest.raises(ConfigError, match="source manifest"):
        load_config(config_file(cfg, tmp_path), image="a.png")


def test_missing_image_rejected_before_cleanup_or_models(cfg, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid selected image reached cleanup or models")

    monkeypatch.setattr(cli, "cleanup_existing_runs", forbidden)
    monkeypatch.setattr(cli, "preflight", forbidden)
    original_handler = signal.getsignal(signal.SIGTERM)
    try:
        assert (
            cli.main(
                [
                    "run",
                    "1",
                    "--config",
                    str(config_file(cfg, tmp_path)),
                    "--gpus",
                    "4,5,6,7",
                    "--image",
                    "missing.png",
                ]
            )
            == 1
        )
        assert not (tmp_path / "runs").exists()
    finally:
        signal.signal(signal.SIGTERM, original_handler)
