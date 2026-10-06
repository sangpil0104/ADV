import json

import pytest

from advv.errors import DataError, ResponseError
from advv.ingest import scan_sources
from advv.sampler import sample_edit, validate_plan
from advv.storage import delete_generated, file_hash, load_rgb, within
from advv.verifiers.parser import parse_response


PROFILE = '"summary":"s","must_preserve":["x"],"uncertain":false'


@pytest.mark.parametrize(
    "text",
    [
        "YES",
        "{" + PROFILE.replace("false", "False") + "}",
        "```json\n{" + PROFILE + "}\n```",
        "{" + PROFILE + "} extra",
        '{"summary":"a",' + PROFILE + "}",
        "{" + PROFILE + ',"extra":1}',
        '{"summary":"s","must_preserve":["x"],"uncertain":NaN}',
        '{"summary":"s","must_preserve":["x"]}',
        "[]",
    ],
)
def test_strict_profile_parser(cfg, text):
    """The source profile is the only parsed model text left (T026); it stays strict."""
    with pytest.raises(ResponseError):
        parse_response(text, cfg["_assets"]["profile_schema"])


def test_profile_parser_accepts_exact_object(cfg):
    assert parse_response(" {" + PROFILE + "} \n", cfg["_assets"]["profile_schema"])["uncertain"] is False


def test_sampler_determinism_and_geometry(cfg, tmp_path):
    source = scan_sources(cfg)[0][0]
    before = file_hash(within(tmp_path / "input", "a.png"))
    profile = {"summary": "A cracked part."}
    left, right = tmp_path / "left", tmp_path / "right"
    for index in range(30):
        a = sample_edit(source, profile, cfg, index, left)
        b = sample_edit(source, profile, cfg, index, right)
        assert a.to_dict() == b.to_dict()
        validate_plan(a, source, left)
        assert a.source_point != a.target_point
    assert before == file_hash(within(tmp_path / "input", "a.png"))


@pytest.mark.parametrize("by", ["group", "pixels"])
def test_split_leakage(cfg, tmp_path, by):
    image = load_rgb(tmp_path / "input/a.png")
    if by == "group":
        image.putpixel((0, 0), (1, 2, 3))
    image.save(tmp_path / "input/b.png")
    rows = [
        {
            "schema_version": "1.1",
            "source_id": "a",
            "image_path": "a.png",
            "split": "train",
            "group_id": "one",
        },
        {
            "schema_version": "1.1",
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


def test_safe_deletion_and_traversal(tmp_path):
    protected = tmp_path / "original.png"
    protected.write_bytes(b"original")
    folder = tmp_path / "candidates/a"
    folder.mkdir(parents=True)
    generated = folder / "generated.png"
    generated.symlink_to(protected)
    with pytest.raises(DataError):
        delete_generated(tmp_path, "a", "candidates/a/generated.png")
    with pytest.raises(DataError):
        delete_generated(tmp_path, "a", "original.png")
    with pytest.raises(DataError):
        within(tmp_path, "../original.png")
    assert protected.read_bytes() == b"original"


def test_input_pixel_dedup(cfg, tmp_path):
    load_rgb(tmp_path / "input/a.png").save(tmp_path / "input/duplicate.bmp")
    sources, stats = scan_sources(cfg)
    assert len(sources) == 1
    assert len(stats["duplicates"]) == 1
