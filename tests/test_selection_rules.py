import copy

import numpy as np
import pytest
from PIL import Image

from advv.accident_group import build_group, contact_components, select_vehicles
from advv.config import validate_config
from advv.errors import ConfigError
from advv.storage import atomic_image, file_hash, within
from advv.visualization import render_selection

YES = [0.1, 0.2, 0.1, 0.2]
SIZE = (400, 200)


class Scorer:
    def __init__(self, default=YES):
        self.default, self.calls = default, 0

    def score_image(self, image):
        self.calls += 1
        return self.default


def segments_from(root, boxes_or_masks, confidences=None):
    entries, masks = [], []
    for i, item in enumerate(boxes_or_masks):
        if isinstance(item, np.ndarray):
            mask = item
        else:
            mask = np.zeros((SIZE[1], SIZE[0]), bool)
            x0, y0, x1, y1 = item
            mask[y0:y1, x0:x1] = True
        path = f"segments/s/seg_{i:04d}.png"
        atomic_image(within(root, path), Image.fromarray(mask.astype(np.uint8) * 255))
        entries.append(
            {
                "segment_id": f"s_segment_{i:04d}",
                "mask_path": path,
                "mask_sha256": file_hash(within(root, path)),
                "class_id": 3,
                "confidence": (confidences or [0.9] * len(boxes_or_masks))[i],
                "area_pixels": int(mask.sum()),
            }
        )
        masks.append(mask)
    return entries, masks


def run(cfg, root, entries, scorer=None):
    image = Image.new("RGB", SIZE, (50, 60, 70))
    scorer = scorer or Scorer()
    return select_vehicles(image, entries, root, cfg, scorer), scorer


def by_id(selection):
    return {d["segment_id"][-4:]: d for d in selection["decisions"]}


def test_example_uses_v2_rules(cfg):
    assert cfg["accident_selection"]["method"] == "clip_contact_context_v2"
    assert cfg["_assets"]["accident_selection"]["version"] == "clip_contact_context_v2"


def test_ego_hood_is_excluded_and_does_not_bridge(cfg, tmp_path):
    # Two touching crash cars; the hood (wide, large, near the bottom edge) touches both.
    entries, _ = segments_from(tmp_path, [(100, 80, 160, 120), (160, 85, 220, 125), (20, 118, 390, 196)])
    selection, scorer = run(cfg, tmp_path, entries)
    d = by_id(selection)
    assert d["0002"]["excluded_reason"] == "ego_vehicle_hood" and not d["0002"]["selected"]
    assert d["0000"]["selected"] and d["0001"]["selected"]
    assert scorer.calls == 2  # one contact unit x two views; the hood is never scored
    assert selection["associations"][0]["member_ids"] == ["s_segment_0000", "s_segment_0001"]
    group = build_group("s", SIZE, entries, selection, tmp_path)
    assert group["member_ids"] == ["s_segment_0000", "s_segment_0001"]


def test_large_vehicle_not_reaching_bottom_is_not_a_hood(cfg, tmp_path):
    entries, _ = segments_from(tmp_path, [(20, 60, 390, 160)])  # 20% gap to the bottom edge
    selection, _ = run(cfg, tmp_path, entries)
    assert by_id(selection)["0000"]["selected"]


def test_duplicate_masks_merge_into_one_vehicle(cfg, tmp_path):
    entries, masks = segments_from(
        tmp_path, [(100, 60, 200, 140), (105, 62, 205, 142), (200, 70, 260, 140)], confidences=[0.7, 0.9, 0.9]
    )
    selection, _ = run(cfg, tmp_path, entries)
    d = by_id(selection)
    assert d["0000"]["duplicate_of"] == "s_segment_0001" and "duplicate_of" not in d["0001"]
    (association,) = selection["associations"]
    assert association["vehicles"] == [["s_segment_0001", "s_segment_0000"], ["s_segment_0002"]]
    assert association["caption_pair"] == "contact_group"
    group = build_group("s", SIZE, entries, selection, tmp_path)
    assert group["area_pixels"] == int((masks[0] | masks[1] | masks[2]).sum())  # union kept, no pixels lost


def test_single_vehicle_with_duplicate_uses_single_captions(cfg, tmp_path):
    entries, _ = segments_from(tmp_path, [(100, 60, 200, 140), (102, 61, 201, 141)])
    selection, _ = run(cfg, tmp_path, entries)
    (association,) = selection["associations"]
    assert association["caption_pair"] == "single" and len(association["vehicles"]) == 1


def test_distant_small_vehicle_overlapping_in_perspective_is_detached(cfg, tmp_path):
    crash = np.zeros((SIZE[1], SIZE[0]), bool)
    crash[60:180, 60:300] = True  # near crash car, bottom 180, bbox height 120
    far = np.zeros_like(crash)
    far[62:75, 250:290] = True  # far car mostly hidden behind/inside the crash mask
    entries, _ = segments_from(tmp_path, [crash, far])
    selection, scorer = run(cfg, tmp_path, entries)
    d = by_id(selection)
    assert d["0001"]["excluded_reason"] == "distant_perspective_overlap" and not d["0001"]["selected"]
    assert d["0001"]["distant"]["anchor_segment_id"] == "s_segment_0000"
    assert d["0001"]["distant"]["overlap_fraction_of_own_area"] > 0.5
    assert d["0000"]["selected"]
    (association,) = selection["associations"]
    assert association["caption_pair"] == "single" and scorer.calls == 2


def test_small_vehicle_at_same_depth_stays_in_contact_unit(cfg, tmp_path):
    # Motorcycle-sized vehicle touching a car at the same ground line is not "distant".
    entries, _ = segments_from(tmp_path, [(60, 60, 300, 180), (300, 160, 315, 178)])
    selection, _ = run(cfg, tmp_path, entries)
    d = by_id(selection)
    assert d["0000"]["selected"] and d["0001"]["selected"]
    assert "excluded_reason" not in d["0001"]


def test_distant_vehicle_bridge_split_rechecks_components(cfg, tmp_path):
    # Far car touches two separate near cars; removing it splits them into independent units.
    entries, _ = segments_from(tmp_path, [(40, 60, 150, 180), (170, 60, 280, 180), (145, 60, 175, 68)])
    selection, scorer = run(cfg, tmp_path, entries)
    d = by_id(selection)
    assert d["0002"]["excluded_reason"] == "distant_perspective_overlap"
    assert [a["caption_pair"] for a in selection["associations"]] == ["single", "single"]
    assert scorer.calls == 4


def test_relative_gap_joins_thin_gap_only_in_v2(cfg, tmp_path):
    _, masks = segments_from(tmp_path, [(100, 60, 200, 140), (200, 60, 300, 140)])
    assert contact_components(masks) == contact_components(masks, 0.0) == [[0, 1]]
    # Two ~200x160 cars (smaller bbox diagonal ~255 px): 0.015 x 255 = 3.8 px tolerance.
    _, far = segments_from(tmp_path, [(0, 20, 200, 180), (204, 20, 400, 180)])  # gap distance 5
    assert contact_components(far, 0.015) == [[0], [1]]
    entries, masks = segments_from(tmp_path, [(0, 20, 200, 180), (202, 20, 400, 180)])  # distance 3
    assert contact_components(masks) == contact_components(masks, 0.0) == [[0], [1]]
    assert contact_components(masks, 0.015) == [[0, 1]]
    selection, _ = run(cfg, tmp_path, entries)
    (association,) = selection["associations"]
    assert association["caption_pair"] == "contact_group"
    old = copy.deepcopy(cfg)
    old["accident_selection"].pop("rules")
    old["accident_selection"]["method"] = "clip_contact_context_v1"
    selection, _ = run(old, tmp_path, entries)
    assert selection["method"] == "clip_contact_context_v1" and len(selection["associations"]) == 2


def test_all_rule_excluded_means_no_group_no_fallback(cfg, tmp_path):
    entries, _ = segments_from(tmp_path, [(20, 118, 390, 196)])
    selection, scorer = run(cfg, tmp_path, entries)
    assert scorer.calls == 0 and not any(d["selected"] for d in selection["decisions"])
    assert build_group("s", SIZE, entries, selection, tmp_path) is None


def test_v2_preview_lists_reasons_and_limits(cfg, tmp_path):
    entries, _ = segments_from(tmp_path, [(100, 80, 160, 120), (20, 118, 390, 196)])
    atomic_image(tmp_path / "source.png", Image.new("RGB", SIZE))
    selection, _ = run(cfg, tmp_path, entries)
    assert "Undetected" in selection["limitations"]
    path = render_selection(tmp_path, {"source_id": "s", "image_path": "source.png"}, entries, selection)
    assert within(tmp_path, path).is_file()


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.pop("rules"),
        lambda s: s["rules"].pop("duplicate_min_iou"),
        lambda s: s["rules"].update(extra=1),
        lambda s: s["rules"].update(duplicate_min_iou=0.0),
        lambda s: s["rules"].update(ego_min_width_fraction=float("nan")),
        lambda s: s["rules"].update(contact_gap_fraction=True),
        lambda s: s.update(method="clip_contact_context_v1"),  # v1 with rules / v2 captions
        lambda s: s.update(method="clip_contact_context_v3"),
    ],
)
def test_invalid_selection_rule_config(cfg, change):
    change(cfg["accident_selection"])
    with pytest.raises(ConfigError):
        validate_config(cfg)


class PerCall:
    def __init__(self, scores):
        self.scores = iter(scores)

    def score_image(self, image):
        return next(self.scores)


def test_separate_tiny_far_unit_passing_clip_is_not_merged_into_group(cfg, tmp_path):
    # Core crash pair near the camera, a separate pair of tiny far cars that CLIP also passes,
    # and a separate same-size vehicle at a similar depth that must stay.
    entries, _ = segments_from(
        tmp_path,
        [
            (40, 90, 150, 185),
            (150, 95, 250, 185),
            (300, 40, 310, 47),
            (310, 41, 320, 48),
            (280, 100, 380, 182),
        ],
    )
    selection, _ = run(cfg, tmp_path, entries, PerCall([YES] * 6))
    d = by_id(selection)
    assert [k for k, v in d.items() if v["selected"]] == ["0000", "0001", "0004"]
    assert d["0002"]["excluded_reason"] == d["0003"]["excluded_reason"] == "distant_separate_unit"
    far = next(a for a in selection["associations"] if "s_segment_0002" in a["member_ids"])
    assert far["clip_passed"] and not far["selected"]
    group = build_group("s", SIZE, entries, selection, tmp_path)
    assert group["member_ids"] == ["s_segment_0000", "s_segment_0001", "s_segment_0004"]
