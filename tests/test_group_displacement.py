import copy
import math

import numpy as np
import pytest
from PIL import Image

from advv.accident_group import build_group, selection_identity
from advv.config import validate_config
from advv.errors import ConfigError, DataError
from advv.group_geometry import group_grid_geometry, translation_fits
from advv.ingest import scan_sources
from advv.sampler import (
    angle_intervals,
    displacement_floor,
    group_diagonal,
    motion_feasible,
    motion_space,
    polar_displacement,
    sample_edit,
    sample_radius,
    source_motion_feasible,
    validate_plan,
)
from advv.segmentation import load_mask
from advv.storage import atomic_image, file_hash, within


def group_from_boxes(cfg, root, boxes):
    source = scan_sources(cfg)[0][0]
    entries, union = [], np.zeros((source.height, source.width), dtype=bool)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        mask = np.zeros_like(union)
        mask[y0:y1, x0:x1] = True
        union |= mask
        path = f"segments/{source.source_id}/car_{i}.png"
        atomic_image(within(root, path), Image.fromarray(mask.astype(np.uint8) * 255))
        entries.append(
            {
                "segment_id": f"car_{i}",
                "class_id": 3,
                "mask_path": path,
                "mask_sha256": file_hash(within(root, path)),
                "area_pixels": int(mask.sum()),
            }
        )
    selection = {
        "identity": selection_identity(cfg),
        "decisions": [{"segment_id": e["segment_id"], "selected": True} for e in entries],
    }
    group = build_group(source.source_id, (source.width, source.height), entries, selection, root)
    return source, {"segments": entries, "selection": selection, "accident_group": group}, union


def plans(source, segments, cfg, root, count=40):
    return [sample_edit(source, segments, cfg, i, root) for i in range(count)]


def test_cap_floor_vertical_scale_and_record(cfg, tmp_path):
    source, segments, union = group_from_boxes(cfg, tmp_path, [(18, 20, 40, 40), (38, 30, 70, 50)])
    diagonal = math.hypot(70 - 18, 50 - 20)
    assert group_diagonal(union) == diagonal
    s = cfg["sampler"]
    low, high = displacement_floor(s, diagonal, (120, 80)), s["max_displacement_fraction"] * diagonal
    assert low == 0.08 * diagonal > 0.03 * math.hypot(120, 80)
    for plan in plans(source, segments, cfg, tmp_path):
        delta = plan.operation_params["displacement_pixels"]
        rec = plan.operation_params["displacement_sampling"]
        assert rec["group_bbox_diagonal_pixels"] == diagonal
        assert rec["min_radius_pixels"] == low and rec["max_radius_pixels"] == high
        assert rec["vertical_scale"] == 0.5 and rec["magnitude_distribution"] == "uniform_radius"
        assert low <= rec["radius_pixels"] <= high and 0 <= rec["angle_radians"] < 2 * math.pi
        assert delta == [
            round(rec["radius_pixels"] * math.cos(rec["angle_radians"])),
            round(0.5 * rec["radius_pixels"] * math.sin(rec["angle_radians"])),
        ]
        assert low <= math.hypot(*delta) <= high and rec["displacement_norm_pixels"] == math.hypot(*delta)
        assert rec["image_diagonal_pixels"] == math.hypot(120, 80)
        assert rec["min_displacement_image_diag_fraction"] == 0.03
        assert abs(delta[1]) <= 0.5 * high + 0.5
        assert translation_fits(union, delta)
        assert plan.sampler_version == "accident_group_v3"


def test_vertical_scale_shrinks_only_dy():
    assert polar_displacement(10.0, math.pi / 2, 1.0) == [0, 10]
    assert polar_displacement(10.0, math.pi / 2, 0.5) == [0, 5]
    assert polar_displacement(10.0, 0.0, 0.5) == [10, 0]
    assert polar_displacement(10.0, math.pi, 0.25) == [-10, 0]


class FixedRng:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


@pytest.mark.parametrize("distribution", ["uniform_radius", "uniform_area"])
def test_radius_distribution_bounds(distribution):
    assert sample_radius(FixedRng(0.0), 2.0, 8.0, distribution) == 2.0
    assert math.isclose(sample_radius(FixedRng(1.0), 2.0, 8.0, distribution), 8.0)
    mid = sample_radius(FixedRng(0.5), 2.0, 8.0, distribution)
    assert mid == (5.0 if distribution == "uniform_radius" else math.sqrt((4 + 64) / 2))
    radii = [sample_radius(np.random.default_rng(i), 2.0, 8.0, distribution) for i in range(200)]
    assert all(2.0 <= r <= 8.0 for r in radii)


def test_uniform_area_config_and_determinism(cfg, tmp_path):
    source, segments, _ = group_from_boxes(cfg, tmp_path, [(30, 20, 70, 50)])
    first = [p.to_dict() for p in plans(source, segments, cfg, tmp_path, 8)]
    assert first == [p.to_dict() for p in plans(source, segments, cfg, tmp_path, 8)]
    other = copy.deepcopy(cfg)
    other["run"]["seed"] = 7
    assert first != [p.to_dict() for p in plans(source, segments, other, tmp_path, 8)]
    area = copy.deepcopy(cfg)
    area["sampler"]["magnitude_distribution"] = "uniform_area"
    validate_config(area)
    for plan in plans(source, segments, area, tmp_path, 8):
        assert plan.operation_params["displacement_sampling"]["magnitude_distribution"] == "uniform_area"


def test_large_group_cap_beyond_frame_room_resamples_inside(cfg, tmp_path):
    # Cap (~17.5 px at the 0.15 default) exceeds the 10 px room on every side; only fitting draws survive.
    source, segments, union = group_from_boxes(cfg, tmp_path, [(10, 10, 60, 70), (58, 10, 110, 70)])
    assert cfg["sampler"]["max_displacement_fraction"] * group_diagonal(union) > 15
    for plan in plans(source, segments, cfg, tmp_path, 10):
        delta = plan.operation_params["displacement_pixels"]
        assert translation_fits(union, delta) and all(abs(v) <= 10 for v in delta)


def test_edge_group_offscreen_draws_are_resampled(cfg, tmp_path):
    source, segments, union = group_from_boxes(cfg, tmp_path, [(80, 30, 118, 60)])
    for plan in plans(source, segments, cfg, tmp_path, 20):
        delta = plan.operation_params["displacement_pixels"]
        assert delta[0] <= 2 and translation_fits(union, delta)


def test_small_group_moves_stay_within_small_cap(cfg, tmp_path):
    cfg["sampler"]["min_displacement_image_diag_fraction"] = 0.0
    source, segments, union = group_from_boxes(cfg, tmp_path, [(50, 30, 62, 42)])
    high = cfg["sampler"]["max_displacement_fraction"] * group_diagonal(union)
    low = cfg["sampler"]["min_displacement_fraction"] * group_diagonal(union)
    assert high < 3
    for plan in plans(source, segments, cfg, tmp_path, 10):
        assert low <= math.hypot(*plan.operation_params["displacement_pixels"]) <= high


def test_small_group_whose_image_floor_exceeds_cap_is_held(cfg, tmp_path):
    # Image floor 0.03 x 144.2 = 4.33 px > cap 0.15 x 17 = 2.55 px: held, the cap is never raised.
    source, segments, union = group_from_boxes(cfg, tmp_path, [(50, 30, 62, 42)])
    space = space_for(cfg, tmp_path, segments, union, source)
    assert math.isclose(space.low, 0.03 * math.hypot(120, 80)) and space.low > space.high
    assert not motion_feasible(space) and not source_motion_feasible(source, segments, cfg, tmp_path)
    with pytest.raises(DataError, match="No valid nonidentity.*floor"):
        sample_edit(source, segments, cfg, 0, tmp_path)


@pytest.mark.parametrize(
    "fraction, image_fraction, expected",
    [(0.08, 0.0, 0.08 * 50), (0.0, 0.03, 0.03 * 500), (0.08, 0.03, 0.03 * 500), (0.4, 0.03, 0.4 * 50)],
)
def test_displacement_floor_is_max_of_group_and_image_terms(fraction, image_fraction, expected):
    settings = {"min_displacement_fraction": fraction, "min_displacement_image_diag_fraction": image_fraction}
    assert math.isclose(displacement_floor(settings, 50.0, (300, 400)), expected)


def test_floor_applies_after_vertical_scale(cfg, tmp_path):
    # Vertical-only room: |delta| = 0.5 r <= 0.5 cap = 0.075 D < floor 0.08 D, so no valid move.
    source, segments, union = group_from_boxes(cfg, tmp_path, [(0, 20, 120, 50)])
    space = space_for(cfg, tmp_path, segments, union, source)
    assert space.scale * space.high < space.low <= space.high
    assert not source_motion_feasible(source, segments, cfg, tmp_path)
    # A free group never gets a rounded |delta| below the floor, including near-vertical draws.
    source, segments, union = group_from_boxes(cfg, tmp_path, [(40, 25, 80, 55)])
    space = space_for(cfg, tmp_path, segments, union, source)
    norms = []
    for plan in plans(source, segments, cfg, tmp_path, 60):
        delta = plan.operation_params["displacement_pixels"]
        rec = plan.operation_params["displacement_sampling"]
        norms.append(math.hypot(*delta))
        assert math.hypot(*delta) >= space.low and math.hypot(*delta) >= rec["min_radius_pixels"]
    assert min(norms) < space.low / space.scale  # The floor is on |delta|, not on r alone.


@pytest.mark.parametrize(
    "boxes",
    [
        [(50, 30, 53, 33)],  # Cap below one feature-grid cell: every draw is identity on the grid.
        [(0, 1, 120, 79)],  # Full-width group: no in-frame nonidentity move.
    ],
)
def test_no_valid_capped_move_keeps_explicit_error(cfg, tmp_path, boxes):
    source, segments, _ = group_from_boxes(cfg, tmp_path, boxes)
    assert not source_motion_feasible(source, segments, cfg, tmp_path)
    with pytest.raises(DataError, match="No valid nonidentity"):
        sample_edit(source, segments, cfg, 0, tmp_path)


def test_tampered_sampling_record_or_delta_fails_validation(cfg, tmp_path):
    source, segments, _ = group_from_boxes(cfg, tmp_path, [(30, 20, 70, 50)])
    plan = sample_edit(source, segments, cfg, 0, tmp_path)
    validate_plan(plan, source, tmp_path)
    for key, value in [
        ("radius_pixels", 1e6),
        ("max_radius_pixels", 1e6),
        ("vertical_scale", 0.0),
        ("angle_radians", 7.0),
        ("magnitude_distribution", "free"),
    ]:
        bad = copy.deepcopy(plan)
        bad.operation_params["displacement_sampling"][key] = value
        with pytest.raises(DataError):
            validate_plan(bad, source, tmp_path)
    bad = copy.deepcopy(plan)
    del bad.operation_params["displacement_sampling"]
    with pytest.raises(DataError, match="sampling record"):
        validate_plan(bad, source, tmp_path)
    bad = copy.deepcopy(plan)
    bad.sampler_version = "accident_group_v1"
    with pytest.raises(DataError):
        validate_plan(bad, source, tmp_path)


@pytest.mark.parametrize(
    "key, value",
    [
        ("version", "accident_group_v1"),
        ("version", "accident_group_v2"),
        ("max_displacement_fraction", 0.0),
        ("max_displacement_fraction", 1.5),
        ("max_displacement_fraction", float("nan")),
        ("max_displacement_fraction", True),
        ("min_displacement_fraction", -0.1),
        ("min_displacement_fraction", 0.3),
        ("min_displacement_image_diag_fraction", -0.01),
        ("min_displacement_image_diag_fraction", 0.6),
        ("min_displacement_image_diag_fraction", float("inf")),
        ("vertical_scale", 0.0),
        ("vertical_scale", 1.2),
        ("magnitude_distribution", "gaussian"),
    ],
)
def test_invalid_displacement_config(cfg, key, value):
    bad = copy.deepcopy(cfg)
    bad["sampler"][key] = value
    with pytest.raises(ConfigError):
        validate_config(bad)
    missing = copy.deepcopy(cfg)
    del missing["sampler"][key]
    with pytest.raises(ConfigError):
        validate_config(missing)


def space_for(cfg, root, segments, union, source):
    size = (source.width, source.height)
    members = [
        load_mask(root, m["mask_path"], m["mask_sha256"], size) for m in segments["accident_group"]["members"]
    ]
    return motion_space(union, members, cfg["sampler"])


def test_full_width_group_samples_inside_feasible_angles(cfg, tmp_path):
    # Zero horizontal room: only near-vertical angles are valid (old uniform-theta draws mostly failed).
    # vertical_scale 1 so the vertical |delta| can reach the floor (see test_floor_applies_after_vertical_scale).
    cfg["sampler"]["vertical_scale"] = 1.0
    source, segments, union = group_from_boxes(cfg, tmp_path, [(0, 20, 120, 50)])
    assert source_motion_feasible(source, segments, cfg, tmp_path)
    space = space_for(cfg, tmp_path, segments, union, source)
    for plan in plans(source, segments, cfg, tmp_path, 40):
        rec = plan.operation_params["displacement_sampling"]
        delta = plan.operation_params["displacement_pixels"]
        starts, ends = angle_intervals(space, rec["radius_pixels"])
        assert any(a <= rec["angle_radians"] <= b for a, b in zip(starts, ends))
        assert math.isclose(rec["feasible_angle_radians"], float((ends - starts).sum()))
        assert rec["feasible_angle_radians"] < math.pi  # Only a small part of the circle survives.
        assert delta[0] == 0 and delta[1] != 0 and translation_fits(union, delta)


def test_angle_intervals_match_pointwise_geometry(cfg, tmp_path):
    source, segments, union = group_from_boxes(cfg, tmp_path, [(70, 10, 118, 40), (60, 38, 90, 60)])
    space = space_for(cfg, tmp_path, segments, union, source)
    members = [union]
    for radius in (space.low, (space.low + space.high) / 2, space.high):
        starts, ends = angle_intervals(space, radius)
        for theta in np.linspace(0, 2 * math.pi, 721, endpoint=False):
            delta = polar_displacement(radius, float(theta), space.scale)
            inside = any(a < theta < b for a, b in zip(starts, ends))
            edge = any(math.isclose(theta, x, abs_tol=1e-9) for x in np.concatenate([starts, ends]))
            if edge:
                continue
            try:
                group_grid_geometry(union, delta, members)
                valid = space.low <= math.hypot(*delta) <= space.high
            except DataError:
                valid = False
            assert inside == valid


@pytest.mark.parametrize(
    "boxes, expected",
    [
        ([(30, 20, 70, 50)], True),
        ([(0, 20, 120, 50)], False),  # Vertical-only room: 0.5 x cap < floor.
        ([(0, 1, 120, 79)], False),
        ([(50, 30, 53, 33)], False),
        ([(0, 0, 120, 80)], False),
    ],
)
def test_motion_feasible_matches_radius_scan(cfg, tmp_path, boxes, expected):
    source, segments, union = group_from_boxes(cfg, tmp_path, boxes)
    space = space_for(cfg, tmp_path, segments, union, source)
    scan = any(len(angle_intervals(space, r)[0]) for r in np.linspace(space.low, space.high, 2001)[1:-1])
    assert motion_feasible(space) is expected and scan is expected


def test_sampling_record_checked_against_ranges_and_run_settings(cfg, tmp_path):
    source, segments, _ = group_from_boxes(cfg, tmp_path, [(30, 20, 70, 50)])
    plan = sample_edit(source, segments, cfg, 0, tmp_path)
    validate_plan(plan, source, tmp_path, cfg["sampler"])
    rec = plan.operation_params["displacement_sampling"]
    diagonal = rec["group_bbox_diagonal_pixels"]
    bad = copy.deepcopy(plan)
    bad.operation_params["displacement_sampling"].update(
        max_displacement_fraction=5.0,
        max_radius_pixels=5.0 * diagonal,
        min_displacement_fraction=-1.0,
        min_radius_pixels=-diagonal,
    )
    with pytest.raises(DataError, match="out of range"):
        validate_plan(bad, source, tmp_path)
    for key, value in [
        ("max_displacement_fraction", 0.25),
        ("min_displacement_fraction", 0.01),
        ("min_displacement_image_diag_fraction", 0.0),
        ("vertical_scale", 1.0),
        ("magnitude_distribution", "uniform_area"),
    ]:
        other = copy.deepcopy(cfg["sampler"])
        other[key] = value
        with pytest.raises(DataError, match="differ from the run sampler config"):
            validate_plan(plan, source, tmp_path, other)
    for key, value in [
        ("feasible_angle_radians", 1.0),
        ("image_diagonal_pixels", 100.0),
        ("min_displacement_image_diag_fraction", 0.9),
    ]:
        bad = copy.deepcopy(plan)
        bad.operation_params["displacement_sampling"][key] = value
        with pytest.raises(DataError):
            validate_plan(bad, source, tmp_path)


def test_recorded_delta_below_floor_fails_validation(cfg, tmp_path):
    source, segments, _ = group_from_boxes(cfg, tmp_path, [(30, 20, 70, 50)])
    plan = sample_edit(source, segments, cfg, 0, tmp_path)
    rec = plan.operation_params["displacement_sampling"]
    # Raising the recorded floor above the actual |delta| (with consistent radius fields) must fail.
    bad = copy.deepcopy(plan)
    bad.operation_params["displacement_sampling"].update(
        min_displacement_image_diag_fraction=(rec["displacement_norm_pixels"] + 0.5)
        / rec["image_diagonal_pixels"],
    )
    with pytest.raises(DataError):
        validate_plan(bad, source, tmp_path)


def test_footer_operation_params_are_short():
    from advv.visualization import params_summary

    text = params_summary(
        {
            "displacement_pixels": [12, -3],
            "group_grid_geometry": {"grid_delta": [6, -2], "grid_size": [56, 40]},
            "displacement_sampling": {
                "magnitude_distribution": "uniform_radius",
                "radius_pixels": 12.345678901234567,
                "angle_radians": 6.0123456789012345,
                "vertical_scale": 0.5,
                "min_radius_pixels": 1.4142135623730951,
                "max_radius_pixels": 14.142135623730951,
                "min_displacement_fraction": 0.03,
                "min_displacement_image_diag_fraction": 0.03,
                "max_displacement_fraction": 0.3,
                "group_bbox_diagonal_pixels": 47.14045207910317,
                "image_diagonal_pixels": 144.22205101855957,
                "displacement_norm_pixels": 12.36931687685298,
                "feasible_angle_radians": 3.141592653589793,
            },
        }
    )
    assert "shift=[12, -3] px" in text and "grid shift=[6, -2]" in text and "r=12.3 px" in text
    assert "cap 1.4-14.1 px" in text and "D=47" in text and "floor>=0.03 x img diag" in text
    assert "12.345678" not in text and "47.140452" not in text and len(text) < 200


def full_frame_backend(cfg, held_names):
    from pathlib import Path

    from conftest import FakeBackend

    class Backend(FakeBackend):
        def segment(self, source, run_dir):
            result = super().segment(source, run_dir)
            if Path(source.original_path).stem not in held_names:
                return result
            mask = np.zeros((source.height, source.width), dtype=np.uint8)
            mask[1:79, :] = 255  # Full width and almost full height: no capped in-frame move.
            path = f"segments/{source.source_id}/full.png"
            atomic_image(within(run_dir, path), Image.fromarray(mask))
            entry = {
                "segment_id": source.source_id + "_car",
                "mask_path": path,
                "mask_sha256": file_hash(within(run_dir, path)),
                "class_id": 3,
            }
            result["segments"] = [entry]
            result["accident_group"] = build_group(
                source.source_id, (source.width, source.height), [entry], result["selection"], run_dir
            )
            return result

    return Backend


def two_source_cfg(cfg, target):
    from pathlib import Path

    Image.new("RGB", (120, 80), (90, 20, 20)).save(Path(cfg["dataset"]["input_dir"]) / "b.png")
    cfg["run"]["target_count"] = target
    return cfg


def test_unmovable_source_is_held_before_generation_and_on_resume(cfg):
    from advv.pipeline import Pipeline, create_run
    from advv.storage import read_json

    cfg = two_source_cfg(cfg, 2)
    Backend = full_frame_backend(cfg, {"a"})
    root = create_run(cfg, "held", provenance="fake")
    first = Backend(cfg, outcomes=[True, KeyboardInterrupt()])
    assert Pipeline(root, first).run()["status"] == "interrupted"
    held_id = next(
        s["source_id"] for s in read_json(root / "sources.json") if s["original_path"].endswith("a.png")
    )
    state = read_json(root / "state.json")
    assert state["held_sources"] == {held_id: "no_valid_motion"}
    assert first.generations and all(g["source_id"] != held_id for g in first.generations)
    second = Backend(cfg)
    assert Pipeline(root, second).run()["status"] == "completed"
    assert all(g["source_id"] != held_id for g in second.generations)
    report = read_json(root / "report.json")
    assert report["held_sources"] == {held_id: "no_valid_motion"}
    assert report["per_source"][held_id] == {"attempts": 0, "accepted": 0, "held": "no_valid_motion"}
    assert "Held sources" in (root / "report.md").read_text()


def test_recorded_hold_persists_even_if_recomputation_differs(cfg):
    from advv.pipeline import Pipeline, create_run
    from advv.storage import atomic_json, read_json
    from conftest import FakeBackend

    cfg = two_source_cfg(cfg, 1)
    root = create_run(cfg, "held_state", provenance="fake")
    sources = [s["source_id"] for s in read_json(root / "sources.json")]
    state = read_json(root / "state.json")
    state["held_sources"] = {sources[0]: "no_valid_motion"}
    atomic_json(root / "state.json", state)
    backend = FakeBackend(cfg)
    assert Pipeline(root, backend).run()["status"] == "completed"
    assert [g["source_id"] for g in backend.generations] == [sources[1]]


def test_all_sources_held_stops_with_explicit_error(cfg):
    from advv.pipeline import Pipeline, create_run

    cfg = two_source_cfg(cfg, 1)
    root = create_run(cfg, "all_held", provenance="fake")
    backend = full_frame_backend(cfg, {"a", "b"})(cfg)
    state = Pipeline(root, backend).run()
    assert state["status"] == "failed" and state["last_error"]["code"] == "no_movable_sources"
    assert len(state["held_sources"]) == 2 and backend.generations == [] and backend.calls == []
