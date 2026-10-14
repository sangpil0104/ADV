import numpy as np
import pytest
from PIL import Image

from advv.accident_group import build_group, detection_filter
from advv.errors import DataError
from advv.storage import atomic_image, within
from advv.visualization import render_selection
from test_selection_rules import SIZE, Scorer, as_v2, by_id, run, segments_from

W, H = SIZE


def box(x0, y0, x1, y1):
    mask = np.zeros((H, W), bool)
    mask[y0:y1, x0:x1] = True
    return mask


def reasons(cfg, mask, confidence=0.9):
    return detection_filter(mask, confidence, SIZE, cfg["accident_selection"]["rules"])[0]


def test_example_filter_defaults(cfg):
    rules = cfg["accident_selection"]["rules"]
    assert {k: rules[k] for k in rules if k not in as_v2(cfg)["accident_selection"]["rules"]} == {
        "detection_min_score": 0.7,
        "max_area_fraction": 0.4,
        "min_area_fraction": 0.001,
        "top_touch_margin_fraction": 0.05,
        "top_touch_max_bottom_fraction": 0.6,
        "min_bottom_fraction": 0.1,
        "min_largest_component_fraction": 0.7,
    }


def test_score_boundary(cfg):
    car = box(100, 80, 160, 130)
    assert reasons(cfg, car, 0.7) == []
    assert reasons(cfg, car, 0.6999) == ["low_detection_score"]


def test_area_boundaries(cfg):
    assert reasons(cfg, box(0, 40, 320, 140)) == []  # 32000 / 80000 = 0.4 exactly
    assert reasons(cfg, box(0, 40, 321, 140)) == ["mask_too_large"]
    assert reasons(cfg, box(100, 100, 108, 110)) == []  # 80 px = 0.001 exactly
    assert reasons(cfg, box(100, 100, 107, 110)) == ["mask_too_small"]


def test_top_touch_structure_needs_both_conditions(cfg):
    # top margin 0.05 x 200 = 10 px (strict), max bottom 0.6 x 200 = 120 px (inclusive)
    assert reasons(cfg, box(20, 9, 120, 120)) == ["elevated_top_structure"]
    assert reasons(cfg, box(20, 10, 120, 120)) == []
    assert reasons(cfg, box(20, 9, 120, 121)) == []  # A close bus reaching the top stays.


def test_ground_height_boundary(cfg):
    assert reasons(cfg, box(200, 12, 220, 20)) == ["above_ground_level"]  # bottom 20 = 0.1 x 200
    assert reasons(cfg, box(200, 12, 220, 21)) == []


def test_fragment_boundary_and_8_connectivity(cfg):
    mask = box(100, 100, 170, 110)  # 700 px
    mask[100:110, 180:210] = True  # 300 px apart -> largest piece 0.7
    assert reasons(cfg, mask) == []
    mask[100:101, 210:211] = True  # second piece grows to 301 px: 700 / 1001 < 0.7
    assert reasons(cfg, mask) == ["fragmented_mask"]
    diagonal = box(100, 100, 120, 120)
    diagonal |= box(120, 120, 140, 140)  # touches only at a corner: one 8-connected piece
    _, metrics = detection_filter(diagonal, 0.9, SIZE, cfg["accident_selection"]["rules"])
    assert metrics["largest_component_fraction"] == 1.0


def test_reasons_keep_rule_order(cfg):
    assert reasons(cfg, box(20, 0, 300, 110), 0.5) == ["low_detection_score", "elevated_top_structure"]


def facade_and_pedestrian_scene(root):
    fragments = box(330, 120, 345, 150) | box(360, 120, 372, 150)  # two pieces split by a pedestrian
    return segments_from(
        root,
        [
            (100, 110, 180, 160),  # crash car
            (180, 115, 250, 160),  # crash car in contact
            (10, 3, 160, 112),  # building facade detected as a bus, overlaps the first car
            fragments,  # high score but fragmented
            (380, 120, 392, 150),  # low-score sliver next to it
        ],
        confidences=[0.99, 0.95, 0.51, 0.92, 0.55],
    )


def test_v3_drops_facade_and_fragments_before_contact_and_clip(cfg, tmp_path):
    entries, _ = facade_and_pedestrian_scene(tmp_path)
    selection, scorer = run(cfg, tmp_path, entries)
    d = by_id(selection)
    assert [k for k, v in d.items() if v["selected"]] == ["0000", "0001"]
    assert d["0002"]["excluded_reason"] == "low_detection_score"
    assert d["0002"]["detection_filter"]["reasons"] == ["low_detection_score", "elevated_top_structure"]
    assert d["0003"]["excluded_reason"] == "fragmented_mask"
    assert d["0004"]["excluded_reason"] == "low_detection_score"
    assert d["0000"]["detection_filter"]["reasons"] == [] and "excluded_reason" not in d["0000"]
    (association,) = selection["associations"]  # The facade no longer bridges into the crash unit.
    assert association["member_ids"] == ["s_segment_0000", "s_segment_0001"] and scorer.calls == 2
    assert (
        selection["method"] == "clip_contact_context_v3" and "Detection filters" in selection["limitations"]
    )
    group = build_group("s", SIZE, entries, selection, tmp_path)
    assert group["member_ids"] == ["s_segment_0000", "s_segment_0001"]
    atomic_image(tmp_path / "source.png", Image.new("RGB", SIZE))
    path = render_selection(tmp_path, {"source_id": "s", "image_path": "source.png"}, entries, selection)
    assert within(tmp_path, path).is_file()


def test_v2_keeps_previous_behavior_without_filters(cfg, tmp_path):
    entries, _ = facade_and_pedestrian_scene(tmp_path)
    selection, _ = run(as_v2(cfg), tmp_path, entries, Scorer())
    d = by_id(selection)
    assert all(v["selected"] for v in d.values())  # The blackbox_002 failure mode the filters fix.
    assert not any("detection_filter" in v for v in d.values())


def test_low_score_duplicate_merges_into_confident_vehicle(cfg, tmp_path):
    entries, masks = segments_from(
        tmp_path, [(100, 60, 200, 140), (104, 62, 206, 142)], confidences=[0.75, 0.58]
    )
    selection, _ = run(cfg, tmp_path, entries)
    d = by_id(selection)
    assert d["0000"]["selected"] and d["0001"]["selected"] and d["0001"]["duplicate_of"] == "s_segment_0000"
    group = build_group("s", SIZE, entries, selection, tmp_path)
    assert group["area_pixels"] == int((masks[0] | masks[1]).sum())  # Union kept, no pixels lost.


def test_filtered_duplicate_unit_is_dropped_whole(cfg, tmp_path):
    entries, _ = segments_from(tmp_path, [(100, 60, 200, 140), (104, 62, 206, 142)], confidences=[0.6, 0.58])
    selection, scorer = run(cfg, tmp_path, entries)
    d = by_id(selection)
    assert d["0000"]["excluded_reason"] == d["0001"]["excluded_reason"] == "low_detection_score"
    assert d["0001"]["duplicate_of"] == "s_segment_0000" and scorer.calls == 0
    assert build_group("s", SIZE, entries, selection, tmp_path) is None  # Held, no fallback.


def test_all_filtered_means_no_group_and_no_clip(cfg, tmp_path):
    entries, _ = segments_from(tmp_path, [(10, 3, 160, 105), (200, 12, 220, 20)], confidences=[0.9, 0.9])
    selection, scorer = run(cfg, tmp_path, entries, Scorer())
    assert [d["excluded_reason"] for d in selection["decisions"]] == [
        "elevated_top_structure",
        "above_ground_level",
    ]
    assert scorer.calls == 0 and build_group("s", SIZE, entries, selection, tmp_path) is None


@pytest.mark.parametrize("confidence", [None, float("nan"), "0.9"])
def test_filter_requires_detection_confidence(cfg, tmp_path, confidence):
    entries, _ = segments_from(tmp_path, [(100, 60, 200, 140)])
    entries[0]["confidence"] = confidence
    with pytest.raises(DataError, match="confidence"):
        run(cfg, tmp_path, entries)
