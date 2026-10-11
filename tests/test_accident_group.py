import copy
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from advv.accident_group import build_group, contact_components, select_vehicles, selection_identity
from advv.backends.dragflow import upstream_instruction
from advv.backends.group_adapter import attach_group_masks, group_coordinate_hook
from advv.errors import BackendError, DataError
from advv.group_geometry import feature_grid, group_grid_geometry, resize_region, translate_mask
from advv.ingest import scan_sources
from advv.motion import diagnose_motion
from advv.sampler import sample_edit, validate_plan
from advv.segmentation import load_mask, validate_segments
from advv.storage import atomic_image, file_hash, load_rgb, within
from advv.visualization import render_selection


def group_fixture(cfg, root):
    source = scan_sources(cfg)[0][0]
    masks = [np.zeros((source.height, source.width), dtype=bool) for _ in range(3)]
    masks[0][20:40, 18:38] = True
    masks[0][26:34, 24:32] = False  # Real hole, not a filled contour.
    masks[0][45:50, 20:25] = True  # Disconnected visible part of the same vehicle.
    masks[1][20:45, 75:100] = True
    masks[2][60:68, 50:60] = True  # Uninvolved vehicle is excluded.
    entries = []
    for i, mask in enumerate(masks):
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
        "decisions": [{"segment_id": e["segment_id"], "selected": i < 2} for i, e in enumerate(entries)],
    }
    group = build_group(source.source_id, (source.width, source.height), entries, selection, root)
    result = {"segments": entries, "selection": selection, "accident_group": group}
    return source, result, masks


def test_whole_group_preserves_holes_components_and_excludes_uninvolved(cfg, tmp_path):
    source, result, masks = group_fixture(cfg, tmp_path)
    validate_segments(source, result, tmp_path)
    endpoints = set()
    for attempt in range(10):
        plan = sample_edit(source, result, cfg, attempt, tmp_path)
        actual = load_mask(tmp_path, plan.region_mask_path, plan.mask_sha256, (source.width, source.height))
        np.testing.assert_array_equal(actual, masks[0] | masks[1])
        assert not actual[29, 27] and actual[47, 22] and not actual[64, 55]
        assert plan.area_pixels == (masks[0] | masks[1]).sum()
        assert not actual[round(plan.source_point[1]), round(plan.source_point[0])]
        endpoints.add(tuple(plan.target_point))
        delta = plan.operation_params["displacement_pixels"]
        moved = translate_mask(actual, *delta)
        np.testing.assert_array_equal(
            moved, translate_mask(masks[0], *delta) | translate_mask(masks[1], *delta)
        )
        assert moved.sum() == actual.sum()
    assert len(endpoints) > 1


def test_group_omitted_member_partial_mask_and_forged_selection_fail(cfg, tmp_path):
    source, result, masks = group_fixture(cfg, tmp_path)
    changed = copy.deepcopy(result)
    changed["accident_group"]["members"].pop()
    with pytest.raises(DataError):
        validate_segments(source, changed, tmp_path)
    changed = copy.deepcopy(result)
    changed["selection"]["decisions"][2]["selected"] = True
    with pytest.raises(DataError, match="selection changed"):
        validate_segments(source, changed, tmp_path)
    plan = sample_edit(source, result, cfg, 1, tmp_path)
    atomic_image(within(tmp_path, plan.region_mask_path), Image.fromarray(masks[0].astype(np.uint8) * 255))
    plan.mask_sha256 = file_hash(within(tmp_path, plan.region_mask_path))
    with pytest.raises(DataError, match="full accident group"):
        validate_plan(plan, source, tmp_path)


def test_no_feasible_translation_and_vanished_member():
    with pytest.raises(DataError, match="leaves"):
        group_grid_geometry(np.ones((80, 120), bool), [2, 0])
    mask = np.zeros((80, 120), bool)
    mask[10:20, 10:20] = True
    with pytest.raises(DataError, match="empty"):
        group_grid_geometry(mask, [0, 0])
    tiny = np.zeros_like(mask)
    tiny[70, 110] = True
    with pytest.raises(DataError, match="vehicle vanished"):
        group_grid_geometry(mask | tiny, [3, 0], [mask, tiny])


@pytest.mark.parametrize("size", [(120, 80), (1920, 1080), (1919, 1079), (641, 483)])
def test_numpy_feature_geometry_matches_torch(size):
    import torch

    torch.set_num_threads(2)
    mask = np.random.default_rng(3).random((size[1], size[0])) > 0.8
    grid = feature_grid(*size)
    expected = torch.nn.functional.interpolate(
        torch.tensor(mask, dtype=torch.float32)[None, None],
        size=tuple(reversed(grid)),
        mode="bilinear",
        align_corners=False,
    )
    np.testing.assert_allclose(resize_region(mask, grid), expected[0, 0].numpy(), atol=1e-4, rtol=0)


def test_contact_graph_does_not_join_nearby_unrelated_vehicles():
    masks = [np.zeros((40, 60), bool) for _ in range(4)]
    masks[0][4:14, 4:14] = True
    masks[1][10:20, 10:20] = True
    masks[2][20:25, 30:35] = True
    masks[3][23:28, 34:39] = True
    assert contact_components(masks) == [[0, 1], [2, 3]]


class Scorer:
    def __init__(self, scores):
        self.scores = iter(scores)

    def score_image(self, image):
        assert image.mode == "RGB"
        return next(self.scores)


def test_selector_both_views_ties_no_all_vehicle_fallback(cfg, tmp_path):
    source, result, _ = group_fixture(cfg, tmp_path)
    yes, no, tie = [0.1, 0.2, 0.1, 0.2], [0.2, 0.1, 0.2, 0.1], [0.1] * 4
    scores = [yes, yes, yes, no, yes, tie]
    selection = select_vehicles(
        load_rgb(Path(source.original_path)), result["segments"], tmp_path, cfg, Scorer(scores)
    )
    assert [d["selected"] for d in selection["decisions"]] == [True, False, False]
    group = build_group(
        source.source_id, (source.width, source.height), result["segments"], selection, tmp_path
    )
    assert group["member_ids"] == ["car_0"]  # Single-vehicle accidents are valid.
    selection = select_vehicles(
        load_rgb(Path(source.original_path)), result["segments"], tmp_path, cfg, Scorer([no] * 6)
    )
    assert (
        build_group(source.source_id, (source.width, source.height), result["segments"], selection, tmp_path)
        is None
    )
    with pytest.raises(DataError, match="Invalid accident"):
        select_vehicles(
            load_rgb(Path(source.original_path)),
            result["segments"],
            tmp_path,
            cfg,
            Scorer([[float("nan")] * 4]),
        )


def test_source_selection_preview_exists_without_candidates(cfg, tmp_path):
    source, result, _ = group_fixture(cfg, tmp_path)
    source_dict = source.to_dict()
    source_dict["image_path"] = "source.png"
    atomic_image(tmp_path / "source.png", load_rgb(Path(source.original_path)))
    path = render_selection(tmp_path, source_dict, result["segments"], result["selection"])
    assert within(tmp_path, path).is_file()
    assert not (tmp_path / "accepted").exists()


def test_official_group_translation_preserves_both_vehicles_and_hook_restores(cfg, tmp_path, monkeypatch):
    import torch

    torch.set_num_threads(2)
    source, result, masks = group_fixture(cfg, tmp_path)
    plan = sample_edit(source, result, cfg, 1, tmp_path)
    mask = masks[0] | masks[1]
    instruction = upstream_instruction(plan.to_dict())
    support = attach_group_masks(instruction, mask, plan, device="cpu", dtype=torch.float32)
    op = instruction["region_operations"]["0"]
    np.testing.assert_array_equal(op["region"][0, 0].numpy(), mask)
    assert np.all(support[mask]) and support.sum() >= mask.sum()
    upstream = Path(__file__).resolve().parents[1] / "third_party/DragFlow"
    if not (upstream / "framework/dragger_utils.py").is_file():
        pytest.skip("Pinned optional upstream checkout unavailable")
    monkeypatch.chdir(upstream)
    monkeypatch.syspath_prepend(str(upstream / "framework"))
    dashboard = importlib.import_module("dashboard_utils")
    estimator = importlib.import_module("dragger_utils").DynamicRegionEstimator
    module = SimpleNamespace(scale_coordinates=dashboard.scale_coordinates)
    original_scale = module.scale_coordinates
    expected = plan.operation_params["group_grid_geometry"]
    with group_coordinate_hook(module):
        module.scale_coordinates(
            instruction, (source.height, source.width), tuple(reversed(expected["grid_size"]))
        )
    assert module.scale_coordinates is original_scale
    op["region_init"] = torch.nn.functional.interpolate(
        op["region"], size=tuple(reversed(expected["grid_size"])), mode="bilinear", align_corners=False
    )
    # Repeat the real Dragger's centroid overwrite, then the pinned affine implementation.
    op["points_fit"][0] = estimator.compute_centroid(op["region_init"])
    op["points_fit"][1] = op["points_fit"][0].clone()
    assert (op["points_fit"][2] - op["points_fit"][0]).tolist() == expected["grid_delta"]
    op, _ = estimator.estimate_inprocessing_state(op, 1.0, is_last_operationidx=True)
    wanted = translate_mask(op["region_init"][0, 0].numpy(), *expected["grid_delta"])
    np.testing.assert_allclose(op["region_curr"][0, 0].numpy(), wanted, atol=1e-5)
    bad = upstream_instruction(plan.to_dict())
    attach_group_masks(bad, mask, plan, device="cpu", dtype=torch.float32)
    with pytest.raises(BackendError), group_coordinate_hook(module):
        module.scale_coordinates(bad, (source.height, source.width), (10, 10))
    assert module.scale_coordinates is original_scale
    assert not torch.cuda.is_initialized()


def test_motion_diagnostics_find_shift_and_identity_but_never_gate(cfg, tmp_path):
    source, result, masks = group_fixture(cfg, tmp_path)
    rng = np.random.default_rng(11)
    arr = np.zeros((source.height, source.width, 3), dtype=np.uint8)
    union = masks[0] | masks[1]
    arr[union] = rng.integers(0, 256, (int(union.sum()), 3), dtype=np.uint8)
    shifted = np.stack([translate_mask(arr[:, :, c], 6, 4) for c in range(3)], axis=-1)
    image = Image.fromarray(arr)
    for target, expected in ((Image.fromarray(shifted), [6, 4]), (image, [0, 0])):
        diag = diagnose_motion(image, target, result["accident_group"], [6, 4], tmp_path)
        assert diag["affects_acceptance"] is False
        for member in diag["members"]:
            assert member["estimated_displacement_pixels"] == expected
    blank = Image.new("RGB", image.size)
    diag = diagnose_motion(blank, blank, result["accident_group"], [6, 4], tmp_path)
    assert all(m["status"] == "unmeasurable" for m in diag["members"])


def test_detected_vehicles_without_selected_accident_never_generate(cfg):
    from advv.pipeline import Pipeline, create_run
    from conftest import FakeBackend

    class NoAccident(FakeBackend):
        def segment(self, source, run_dir):
            result = super().segment(source, run_dir)
            result["accident_group"] = None
            for decision in result["selection"]["decisions"]:
                decision["selected"] = False
            return result

    root = create_run(cfg, "no_accident", provenance="fake")
    backend = NoAccident(cfg)
    state = Pipeline(root, backend).run()
    assert state["status"] == "failed"
    assert state["last_error"]["code"] == "no_eligible_sources"
    assert backend.generations == [] and backend.calls == []


def test_motion_matcher_error_is_a_diagnostic_not_candidate_rejection(cfg, tmp_path, monkeypatch):
    import cv2
    source, result, _ = group_fixture(cfg, tmp_path)
    arr = np.random.default_rng(4).integers(0, 255, (source.height, source.width, 3), dtype=np.uint8)

    def fail(*args, **kwargs):
        raise cv2.error('fixture')

    monkeypatch.setattr(cv2, 'matchTemplate', fail)
    diag = diagnose_motion(Image.fromarray(arr), Image.fromarray(arr), result['accident_group'], [4, 2], tmp_path)
    assert not diag['affects_acceptance']
    assert all(m['status'] == 'matching_error' for m in diag['members'])
