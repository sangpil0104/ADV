import copy
import itertools
import json

import numpy as np
import pytest

from advv.backends.clip import letterbox
from advv.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config, validate_config
from advv.errors import ConfigError, DataError, ResponseError
from advv.ingest import scan_sources
from advv.sampler import sample_edit, validate_plan
from advv.storage import file_hash, load_rgb, within
from advv.verifiers.decision import decide, evaluate_scores
from conftest import fixture_segments


def test_project_relative_default_paths():
    cfg = load_config(PROJECT_ROOT / "configs/advv.example.yaml", target_count=10, gpus="4,5,6,7")
    assert DEFAULT_CONFIG_PATH == PROJECT_ROOT / "configs/advv.clip.local.yaml"
    assert cfg["dataset"]["root"] == cfg["dataset"]["input_dir"] == str(PROJECT_ROOT / "assets/inputs")
    assert cfg["run"]["output_root"] == str(PROJECT_ROOT / "assets/outputs")


@pytest.mark.parametrize("bits", list(itertools.product([False, True], repeat=5)))
def test_all_32_policy_combinations(bits):
    scores = [s for bit in bits for s in (0.2, 0.3 if bit else 0.1)]
    result = evaluate_scores(scores)
    assert (decide(result) == "accepted") == (sum(bits[:4]) >= 3 and bits[4])
    assert result["physical_pass_count"] == sum(bits[:4])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, None, "0.3", 2.0])
def test_invalid_numeric_scores_never_count_as_one_allowed_failure(value):
    scores = [0.1, 0.2] * 5
    scores[0] = value
    with pytest.raises(ResponseError):
        evaluate_scores(scores)


def test_tie_and_incomplete_scores():
    result = evaluate_scores([0.1, 0.2] * 4 + [0.2, 0.2])
    assert decide(result) == "rejected"
    with pytest.raises(ResponseError):
        evaluate_scores([0.1] * 9)
    result["passed"] = True
    with pytest.raises(ResponseError):
        decide(result)


def test_mask_geometry_determinism_and_prompt_isolation(cfg, tmp_path):
    source = scan_sources(cfg)[0][0]
    before = file_hash(tmp_path / "input/a.png")
    left, right = tmp_path / "left", tmp_path / "right"
    a_segments = fixture_segments(source, left, cfg)
    b_segments = fixture_segments(source, right, cfg)
    endpoint_set, area_set = set(), set()
    for index in range(12):
        a = sample_edit(source, a_segments, cfg, index, left)
        b = sample_edit(source, b_segments, cfg, index, right)
        assert a.to_dict() == b.to_dict()
        validate_plan(a, source, left)
        endpoint_set.add(tuple(a.target_point))
        area_set.add(a.area_pixels)
        assert a.area_pixels == round(a.range_fraction_of_image * source.width * source.height)
    assert len(endpoint_set) > 1 and len(area_set) == 1
    assert file_hash(tmp_path / "input/a.png") == before
    changed = copy.deepcopy(cfg)
    changed["generation"]["augmentation_prompt"] = "Keep the accident, change the vehicle pose."
    changed["generation"]["effective_augmentation_prompt"] = changed["generation"]["augmentation_prompt"]
    c = sample_edit(source, a_segments, changed, 0, left)
    original = sample_edit(source, a_segments, cfg, 0, right)
    assert c.source_point == original.source_point and c.target_point == original.target_point
    assert c.mask_sha256 == original.mask_sha256
    assert changed["_assets"]["criteria_sha256"] == cfg["_assets"]["criteria_sha256"]
    assert changed["generation"]["augmentation_prompt"] in c.target_prompt


@pytest.mark.parametrize("by", ["group", "pixels"])
def test_split_leakage(cfg, tmp_path, by):
    image = load_rgb(tmp_path / "input/a.png")
    if by == "group":
        image.putpixel((0, 0), (1, 2, 3))
    image.save(tmp_path / "input/b.png")
    rows = [
        {
            "schema_version": "2.0",
            "source_id": "a",
            "image_path": "a.png",
            "split": "train",
            "group_id": "one",
        },
        {
            "schema_version": "2.0",
            "source_id": "b",
            "image_path": "b.png",
            "split": "test",
            "group_id": "one" if by == "group" else "two",
        },
    ]
    manifest = tmp_path / "sources.jsonl"
    manifest.write_text("\n".join(map(json.dumps, rows)))
    cfg["dataset"].update(input_dir=None, manifest=str(manifest))
    with pytest.raises(DataError):
        scan_sources(cfg)


def test_input_dedup_and_traversal(cfg, tmp_path):
    load_rgb(tmp_path / "input/a.png").save(tmp_path / "input/duplicate.bmp")
    sources, stats = scan_sources(cfg)
    assert len(sources) == 1 and len(stats["duplicates"]) == 1
    with pytest.raises(DataError):
        within(tmp_path, "../outside")


def test_no_venv_overrides_or_retained_rejects(cfg):
    cfg["execution"]["generator_python"] = "/tmp/.venv/bin/python"
    with pytest.raises(ConfigError):
        validate_config(cfg)
    cfg["execution"].pop("generator_python")
    cfg["storage"]["retain_rejected_details"] = True
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_input_inside_managed_outputs_is_rejected_before_cleanup(cfg):
    cfg["dataset"]["root"] = cfg["run"]["output_root"] + "/old_run/quarantine"
    cfg["dataset"]["input_dir"] = cfg["dataset"]["root"]
    with pytest.raises(ConfigError, match="must not overlap"):
        validate_config(cfg)


def test_prompt_precedence_and_recipe(cfg, tmp_path):
    import yaml
    from advv.config import recipe_hash

    cfg["generation"]["augmentation_prompt"] = "YAML prompt"
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump(cfg))
    loaded = load_config(path, prompt="CLI prompt")
    assert loaded["generation"]["effective_augmentation_prompt"] == "CLI prompt"
    original = load_config(path)
    assert original["generation"]["effective_augmentation_prompt"] == "YAML prompt"
    assert loaded["_assets"] == original["_assets"]
    assert recipe_hash(loaded) != recipe_hash(original)
    with pytest.raises(ConfigError):
        load_config(path, prompt=" ")


def test_whole_frame_letterbox_preserves_both_edges():
    from PIL import Image

    image = Image.new("RGB", (120, 40), "red")
    image.putpixel((0, 20), (0, 255, 0))
    image.putpixel((119, 20), (0, 0, 255))
    boxed = letterbox(image, 120)
    assert boxed.size == (120, 120)
    assert boxed.getpixel((0, 60)) == (0, 255, 0)
    assert boxed.getpixel((119, 60)) == (0, 0, 255)
    assert boxed.getpixel((0, 0)) == (127, 127, 127)
    assert np.asarray(boxed).shape == (120, 120, 3)
