"""CPU contact sheets from committed accepted images; never an inference input."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .errors import DataError
from .storage import (
    atomic_image,
    atomic_json,
    digest,
    file_hash,
    load_rgb,
    pixel_hash,
    read_json,
    safe_id,
    within,
)
from .visualization import _font, render_plan_panel

GALLERY_VERSION = "accepted_grid_v1"
DRAG_PLAN_GALLERY_VERSION = "accepted_drag_plan_photos_v2"


def gallery_records(root: Path):
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema_version") not in ("1.1", "2.0"):
        raise DataError("Unsupported run schema for visualization")
    records = []
    for path in (root / "records").glob("*.json"):
        record = read_json(path)
        if record.get("status") != "accepted" or record.get("export_status") != "eligible":
            continue
        if manifest["schema_version"] == "2.0":
            if not record.get("presentation_complete"):
                continue
            from .verifiers.decision import decide

            if decide(record["clip"]) != "accepted":
                raise DataError("Invalid accepted CLIP record")
        safe_id(record["candidate_id"])
        records.append(record)
    return sorted(records, key=lambda r: (r["sequence"], r["candidate_id"]))


def render_gallery(root: Path, records=None, *, kind="generated"):
    """Caller owns run lock. Refresh on each accepted commit, exit, or CPU rebuild."""
    if kind not in ("generated", "drag_plan"):
        raise DataError(f"Unsupported gallery kind: {kind}")
    is_plan = kind == "drag_plan"
    version = DRAG_PLAN_GALLERY_VERSION if is_plan else GALLERY_VERSION
    title = "DRAG PLANS" if is_plan else "GENERATED IMAGES"
    image_path = f"visualizations/{kind}_grid.png"
    root = root.resolve()
    records = gallery_records(root) if records is None else records
    legacy = read_json(root / "manifest.json")["schema_version"] == "1.1"
    sources = {s["source_id"]: s for s in read_json(root / "sources.json")} if is_plan and records else {}
    settings = read_json(root / "config.json")["visualization"] if is_plan and records else {}
    panel_inputs = {}
    rows = []
    for i, record in enumerate(records, 1):
        relative = record["visualization"]["artifacts"]["drag_plan"] if is_plan else record["image_path"]
        path = within(root, relative, must_exist=True)
        actual_hash = file_hash(path)
        if not is_plan and record.get("file_sha256") and record["file_sha256"] != actual_hash:
            raise DataError("Accepted image changed before gallery rendering")
        rows.append(
            {
                "index": i,
                "candidate_id": record["candidate_id"],
                "source_id": record["source_id"],
                "image_path": relative,
                "file_sha256": actual_hash,
                "verification_status": "legacy_unverified" if legacy else "clip_verified",
            }
        )
        if is_plan:
            source = sources[record["source_id"]]
            metadata = read_json(
                within(root, record["visualization"]["artifacts"]["metadata"], must_exist=True)
            )
            source_path = within(root, source["image_path"], must_exist=True)
            source_hash = file_hash(source_path)
            panel_inputs[record["candidate_id"]] = (source_path, record["plan"], metadata["display_geometry"])
            rows[-1]["panel_identity"] = digest(
                [source_hash, source["pixel_sha256"], record["plan"], metadata["display_geometry"], settings]
            )
    identity = digest([version, root.name, rows])
    metadata_path = within(root, f"visualizations/{kind}_grid.json")
    destination = within(root, image_path)
    if metadata_path.is_file() and destination.is_file():
        cached = read_json(metadata_path)
        if cached.get("identity") == identity and cached.get("image_sha256") == file_hash(destination):
            return cached
    count = len(rows)
    columns = max(1, math.ceil(math.sqrt(count if count <= 16 else count * 0.65)))
    row_count = max(1, math.ceil(count / columns))
    # Keep every image in one bounded overview; full-resolution images remain separate.
    base_width, base_height = (720, 405) if is_plan else (480, 270)
    nominal_width = columns * (base_width + 12) + 12
    nominal_height = row_count * (base_height + 62) + 102
    scale = min(
        1.0,
        8000 / max(nominal_width, nominal_height),
        math.sqrt(32_000_000 / (nominal_width * nominal_height)),
    )
    tile_width = max(1, int(base_width * scale))
    tile_height = max(1, round(tile_width * base_height / base_width))
    gap, heading, caption = max(0, int(12 * scale)), 90, int(50 * scale) if tile_width >= 160 else 0
    if is_plan:
        heading = caption = 0
    width = max(640, columns * (tile_width + gap) + gap)
    height = heading + row_count * (tile_height + caption + gap) + gap
    canvas = Image.new("RGB", (width, height), "#171B24")
    draw = ImageDraw.Draw(canvas)
    if not is_plan:
        draw.text((gap, 12), title + " | " + root.name, font=_font(20), fill="white")
        status = "LEGACY / not CLIP verified" if legacy else "Accepted images / CLIP verified"
        draw.text(
            (gap, 43),
            f"{count} images | {status} | ordered by acceptance sequence",
            font=_font(15),
            fill="#CED5DF",
        )
    for i, row in enumerate(rows):
        x = gap + (i % columns) * (tile_width + gap)
        y = heading + (i // columns) * (tile_height + caption + gap)
        if is_plan:
            y += gap
            source_path, plan, geometry = panel_inputs[row["candidate_id"]]
            original = load_rgb(source_path)
            if pixel_hash(original) != sources[row["source_id"]]["pixel_sha256"]:
                raise DataError("Source image changed before plan gallery rendering")
            frame, _ = render_plan_panel(root, original, plan, geometry, settings, show_labels=False)
        else:
            frame = load_rgb(within(root, row["image_path"], must_exist=True))
        thumb = ImageOps.contain(frame, (tile_width, tile_height), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (x + (tile_width - thumb.width) // 2, y + (tile_height - thumb.height) // 2))
        if caption:
            font = _font(max(8, round(12 * scale)))
            label = f"#{i + 1:03d}  {row['candidate_id']}"
            while len(label) > 8 and draw.textlength(label, font=font) > tile_width - 4:
                label = label[:-2] + "…"
            draw.text((x, y + tile_height + 2), label, font=font, fill="white")
            draw.text(
                (x, y + tile_height + caption // 2),
                f"source: {row['source_id'][:24]}",
                font=font,
                fill="#AFC0D5",
            )
    if not rows and not is_plan:
        draw.text((gap, heading + 20), "No accepted images yet.", font=_font(20), fill="white")
    atomic_image(destination, canvas)
    metadata = {
        "version": version,
        "identity": identity,
        "run_id": root.name,
        "count": count,
        "image_path": image_path,
        "image_sha256": file_hash(destination),
        "size": list(canvas.size),
        "columns": columns,
        "images": rows,
    }
    atomic_json(metadata_path, metadata)
    return metadata
