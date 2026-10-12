import numpy as np
import pytest
from PIL import Image

from advv.config import load_config, validate_config
from advv.errors import ConfigError, DataError
from advv.ingest import downscale
from advv.pipeline import Pipeline, create_run
from advv.reporting import records_for
from advv.segmentation import load_mask
from advv.storage import file_hash, load_rgb, pixel_hash, read_json, within
from conftest import PROJECT, FakeBackend


def large_cfg(tmp_path, size=(400, 200), max_side=200):
    folder = tmp_path / "large"
    folder.mkdir()
    arr = np.random.default_rng(0).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(folder / "big.png")
    cfg = load_config(PROJECT / "configs/advv.example.yaml", input_dir=folder, target_count=1, gpus="0,1")
    cfg["run"]["output_root"] = str(tmp_path / "runs")
    cfg["report"]["terminal_progress"] = False
    cfg["dataset"]["max_long_side_px"] = max_side
    return cfg, folder / "big.png"


def test_example_defaults_downscale_960_and_smaller_cap():
    cfg = load_config(PROJECT / "configs/advv.example.yaml", gpus="0,1")
    assert cfg["dataset"]["max_long_side_px"] == 960
    assert cfg["dataset"]["downscale_resample"] == "lanczos"
    assert cfg["sampler"]["max_displacement_fraction"] == 0.15


@pytest.mark.parametrize(
    "size, expected", [((1920, 1080), (960, 540)), ((1080, 1920), (540, 960)), ((1280, 720), (960, 540))]
)
def test_downscale_long_side_and_record(size, expected):
    image, info = downscale(Image.new("RGB", size, (9, 9, 9)), 960, "lanczos")
    assert image.size == expected and info["resized"]
    assert info["original_size"] == list(size) and info["snapshot_size"] == list(expected)
    assert info["scale"] == [expected[0] / size[0], expected[1] / size[1]]
    assert info["resample"] == "lanczos" and info["library"].startswith("Pillow ")


@pytest.mark.parametrize("limit", [960, None])
def test_small_images_are_never_upscaled(limit):
    source = Image.new("RGB", (640, 480), (1, 2, 3))
    image, info = downscale(source, limit, "lanczos")
    assert image.size == (640, 480) and not info["resized"] and info["scale"] == [1.0, 1.0]
    assert pixel_hash(image) == pixel_hash(source)


def test_downscale_refuses_too_thin_result():
    with pytest.raises(DataError, match="32 pixels"):
        downscale(Image.new("RGB", (4000, 100)), 960, "lanczos")


def test_snapshot_is_downscaled_original_untouched_and_recorded(tmp_path):
    cfg, original = large_cfg(tmp_path)
    before = file_hash(original)
    root = create_run(cfg, "small", provenance="fake")
    assert file_hash(original) == before
    (source,) = read_json(root / "sources.json")
    snapshot = load_rgb(within(root, source["image_path"]))
    assert snapshot.size == (200, 100) == (source["width"], source["height"])
    assert source["pixel_sha256"] == pixel_hash(snapshot) != source["preprocessing"]["original_pixel_sha256"]
    assert source["preprocessing"]["original_size"] == [400, 200]
    assert source["preprocessing"]["scale"] == [0.5, 0.5]
    assert source["file_sha256"] == before
    assert read_json(root / "config.json")["dataset"]["max_long_side_px"] == 200


def test_whole_pipeline_uses_snapshot_grid(tmp_path):
    cfg, _ = large_cfg(tmp_path)
    root = create_run(cfg, "grid", provenance="fake")
    backend = FakeBackend(cfg)
    assert Pipeline(root, backend).run()["status"] == "completed"
    (record,) = records_for(root)
    assert record["source_preprocessing"]["snapshot_size"] == [200, 100]
    plan = record["plan"]
    for x, y in (plan["source_point"], plan["target_point"]):
        assert 0 <= x < 200 and 0 <= y < 100
    mask = load_mask(root, plan["region_mask_path"], plan["mask_sha256"], (200, 100))
    assert mask.shape == (100, 200) and mask.any()
    assert load_rgb(within(root, record["image_path"])).size == (200, 100)
    metadata = read_json(within(root, record["visualization"]["artifacts"]["metadata"]))
    assert metadata["source_preprocessing"]["original_size"] == [400, 200]


def test_recipe_changes_with_resolution_setting(tmp_path):
    from advv.config import recipe_hash

    cfg, _ = large_cfg(tmp_path)
    other = dict(cfg, dataset=dict(cfg["dataset"], max_long_side_px=300))
    assert recipe_hash(cfg) != recipe_hash(other)


@pytest.mark.parametrize(
    "key, value",
    [
        ("max_long_side_px", 32),
        ("max_long_side_px", 960.0),
        ("max_long_side_px", True),
        ("downscale_resample", "nearest"),
    ],
)
def test_invalid_downscale_settings(cfg, key, value):
    cfg["dataset"][key] = value
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_missing_downscale_setting_is_an_error(cfg):
    del cfg["dataset"]["max_long_side_px"]
    with pytest.raises(ConfigError):
        validate_config(cfg)
