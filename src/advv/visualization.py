"""CPU-only, deterministic previews from persisted requested/effective geometry."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont

from .storage import atomic_image, atomic_json, digest, file_hash, load_rgb, safe_id, within
from .errors import DataError

RENDERER_VERSION = "pillow_accident_group_v1"
LAYOUT_VERSION = "by_kind_v1"


def artifact_paths(candidate_id):
    cid = safe_id(candidate_id)
    return {
        "comparison": f"visualizations/comparison/{cid}.png",
        "drag_plan": f"visualizations/drag_plan/{cid}.png",
        "metadata": f"visualizations/metadata/{cid}.json",
    }


def render_selection(run_dir, source, segments, selection):
    """Source inspection, not a rejected candidate visualization."""
    from .segmentation import load_mask

    image = load_rgb(within(run_dir, source["image_path"], must_exist=True))
    factor = min(1.0, 1024 / max(image.size))
    size = tuple(round(v * factor) for v in image.size)
    overlay = image.resize(size, Image.Resampling.LANCZOS).convert("RGBA")
    selected = {d["segment_id"] for d in selection["decisions"] if d["selected"]}
    labels = []
    for entry in segments:
        mask = load_mask(run_dir, entry["mask_path"], entry["mask_sha256"], image.size)
        visible = Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)
        chosen = entry["segment_id"] in selected
        color = "#43D890" if chosen else "#EE6666"
        layer = Image.new("RGBA", size, color)
        layer.putalpha(visible.point(lambda p: round(p * 0.3)))
        overlay = Image.alpha_composite(overlay, layer)
        ys, xs = np.nonzero(mask)
        labels.append(
            (
                float(xs.mean()) * factor,
                float(ys.mean()) * factor,
                entry["segment_id"].split("_segment_")[-1],
                color,
            )
        )
    draw = ImageDraw.Draw(overlay)
    for x, y, label, color in labels:
        draw.text((x, y), label, font=_font(18), fill=color, stroke_width=2, stroke_fill="black")
    lines = [
        "SOURCE SELECTION | green=selected accident vehicle | red=excluded",
        f"Method {selection.get('method', 'unknown')}: automatic CLIP proxy"
        + (" + deterministic rules" if selection.get("rules") else "")
        + "; inspect vehicle IDs. Selection correctness is not guaranteed.",
    ]
    if selection.get("limitations"):
        lines += textwrap.wrap("Limits: " + selection["limitations"], 110)
    for d in selection["decisions"]:
        notes = [d[k] for k in ("excluded_reason",) if d.get(k)]
        f = d.get("detection_filter") or {}
        if f.get("reasons"):
            notes = ["+".join(f["reasons"])]
            notes.append(
                f"score {f['confidence']:.2f}, area {f['area_fraction']:.3f}, "
                f"top {f['bbox_top_fraction']:.2f}, bottom {f['bbox_bottom_fraction']:.2f}, "
                f"largest piece {f['largest_component_fraction']:.2f}"
            )
        if d.get("duplicate_of"):
            notes.append("same vehicle as " + d["duplicate_of"].split("_segment_")[-1])
        suffix = f" ({'; '.join(notes)})" if notes else ""
        lines += textwrap.wrap(f"{'SELECT' if d['selected'] else 'EXCLUDE'} {d['segment_id']}{suffix}", 110)
    canvas = Image.new("RGB", (max(size[0], 900), size[1] + 28 * len(lines)), "#171B24")
    canvas.paste(overlay.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        draw.text((10, size[1] + i * 28), line, font=_font(15), fill="white")
    relative = f"segments/{source['source_id']}/selection.png"
    atomic_image(within(run_dir, relative), canvas)
    return relative


def _font(size=16):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def _status(record, name):
    check = record.get("checks", {}).get(name, {})
    return (
        check.get("response", {}).get("answer")
        if check.get("response")
        else check.get("status", "pending").upper()
    )


def _short(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_short(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}={_short(v)}" for k, v in value.items()) + "}"
    return str(value)


def params_summary(params: dict) -> str:
    """Short footer text; full-precision values stay in the plan/metadata JSON."""
    known = {"displacement_pixels", "group_grid_geometry", "displacement_sampling"}
    parts = []
    if "displacement_pixels" in params:
        parts.append(f"shift={_short(params['displacement_pixels'])} px")
    if isinstance(params.get("group_grid_geometry"), dict) and "grid_delta" in params["group_grid_geometry"]:
        parts.append(f"grid shift={_short(params['group_grid_geometry']['grid_delta'])}")
    s = params.get("displacement_sampling")
    if isinstance(s, dict):
        try:
            floor = s.get("min_displacement_image_diag_fraction")
            parts.append(
                f"r={s['radius_pixels']:.1f} px theta={math.degrees(s['angle_radians']):.1f} deg "
                f"| cap {s['min_radius_pixels']:.1f}-{s['max_radius_pixels']:.1f} px "
                f"({s['min_displacement_fraction']:.4g}-{s['max_displacement_fraction']:.4g} x D={s['group_bbox_diagonal_pixels']:.0f}"
                + ("" if floor is None else f", floor>={floor:.4g} x img diag")
                + f") | vertical_scale={s['vertical_scale']:.4g} {s['magnitude_distribution']}"
            )
        except (KeyError, TypeError, ValueError):
            parts.append(f"sampling={_short(s)}")
    parts.extend(f"{k}={_short(v)}" for k, v in params.items() if k not in known)
    return " | ".join(parts)


def render_plan_panel(run_dir, original, plan, geometry, settings, *, show_labels=True):
    """Render only the photograph and drag geometry; no heading or footer."""
    w, h = original.size
    mask_path = within(run_dir, geometry["region_mask_path"], must_exist=True)
    if geometry.get("mask_sha256") and file_hash(mask_path) != geometry["mask_sha256"]:
        raise DataError("Effective visualization mask changed")
    with Image.open(mask_path) as image:
        mask = image.convert("L").resize((w, h), Image.Resampling.NEAREST)
    area = int(np.count_nonzero(np.asarray(mask)))
    factor = min(1.0, settings["max_panel_side_px"] / max(w, h))
    pw, ph = max(1, round(w * factor)), max(1, round(h * factor))
    # Actual rounded panel sizes determine transform, never a guessed scale.
    sx, sy = pw / w, ph / h
    overlay = original.resize((pw, ph), Image.Resampling.LANCZOS).convert("RGBA")
    preview_mask = mask.resize((pw, ph), Image.Resampling.NEAREST)
    blue = Image.new("RGBA", overlay.size, ImageColor.getrgb(settings["area_color"]) + (0,))
    blue.putalpha(preview_mask.point(lambda x: round(x * settings["area_alpha"])))
    overlay = Image.alpha_composite(overlay, blue)
    if settings["draw_area_outline"]:
        binary = preview_mask.point(lambda x: 255 if x else 0)
        border = np.asarray(binary) - np.asarray(binary.filter(ImageFilter.MinFilter(3)))
        outline = Image.new("RGBA", overlay.size, settings["area_color"])
        outline.putalpha(Image.fromarray(border))
        overlay = Image.alpha_composite(overlay, outline)
    if plan.get("segment_mask_path"):
        with Image.open(within(run_dir, plan["segment_mask_path"], must_exist=True)) as segment:
            binary = segment.convert("L").resize((pw, ph), Image.Resampling.NEAREST)
        border = np.asarray(binary) - np.asarray(binary.filter(ImageFilter.MinFilter(3)))
        outline = Image.new("RGBA", overlay.size, "#FFFFFF")
        outline.putalpha(Image.fromarray(border))
        overlay = Image.alpha_composite(overlay, outline)
    group = plan.get("accident_group")
    if group:
        from .group_geometry import translate_mask

        delta = [round(geometry["target_point"][i] - geometry["source_point"][i]) for i in (0, 1)]
        target = translate_mask(np.asarray(mask) > 0, *delta)
        binary = Image.fromarray(target.astype(np.uint8) * 255).resize((pw, ph), Image.Resampling.NEAREST)
        border = np.asarray(binary) - np.asarray(binary.filter(ImageFilter.MinFilter(3)))
        outline = Image.new("RGBA", overlay.size, "#00FFAA")
        outline.putalpha(Image.fromarray(border))
        overlay = Image.alpha_composite(overlay, outline)
        labels = ImageDraw.Draw(overlay)
        for member in group["members"] if show_labels else []:
            with Image.open(within(run_dir, member["mask_path"], must_exist=True)) as member_image:
                ys, xs = np.nonzero(np.asarray(member_image))
            labels.text(
                (xs.mean() * sx, ys.mean() * sy),
                member["segment_id"].split("_segment_")[-1],
                font=_font(15),
                fill="white",
                stroke_width=2,
                stroke_fill="black",
            )
    draw = ImageDraw.Draw(overlay)
    start = tuple(geometry["source_point"][i] * (sx, sy)[i] for i in range(2))
    end = tuple(geometry["target_point"][i] * (sx, sy)[i] for i in range(2))
    draw.line([start, end], fill="black", width=6)
    draw.line([start, end], fill=settings["arrow_color"], width=3)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = min(12, max(5, math.dist(start, end) * 0.4))
    wings = [
        (end[0] - length * math.cos(angle + offset), end[1] - length * math.sin(angle + offset))
        for offset in (-0.5, 0.5)
    ]
    draw.line([wings[0], end, wings[1]], fill="black", width=6)
    draw.line([wings[0], end, wings[1]], fill=settings["arrow_color"], width=3)
    radius = 6

    def label(x, y, text, direction):
        if not show_labels:
            return
        font = _font(15)
        x = max(2, min(pw - 21, x + 10 * direction))
        y = max(2, min(ph - 23, y - 25 if direction < 0 else y + 8))
        draw.text((x, y), text, fill="white", font=font, stroke_width=2, stroke_fill="black")

    x, y = start
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=settings["start_color"],
        outline="white",
        width=2,
    )
    label(x, y, "S", -1)
    x, y = end
    for a, b in (((x - radius, y), (x + radius, y)), ((x, y - radius), (x, y + radius))):
        draw.line((a, b), fill="black", width=6)
        draw.line((a, b), fill=settings["end_color"], width=3)
    label(x, y, "E", 1)
    anchor = geometry.get("anchor_point")
    if anchor and settings["show_anchor"]:
        x, y = anchor[0] * sx, anchor[1] * sy
        draw.polygon(
            [(x, y - 7), (x + 7, y), (x, y + 7), (x - 7, y)], fill=settings["anchor_color"], outline="white"
        )
        label(x, y, "A", -1)
    return overlay.convert("RGB"), area


def render(run_dir: Path, source: dict, record: dict, settings: dict) -> dict:
    if record.get("status") != "accepted" or not record.get("clip", {}).get("passed"):
        raise DataError("Only verified accepted candidates may have public visualizations")
    if not settings["enabled"]:
        return {"status": "disabled"}
    cid, plan = record["candidate_id"], record["plan"]
    destinations = artifact_paths(cid)
    original = load_rgb(within(run_dir, source["image_path"], must_exist=True))
    w, h = original.size
    effective = record.get("effective")
    geometry = effective or dict(
        source_point=plan["source_point"],
        target_point=plan["target_point"],
        anchor_point=plan.get("anchor_point"),
        region_mask_path=plan["region_mask_path"],
    )
    overlay, area = render_plan_panel(run_dir, original, plan, geometry, settings)
    pw, ph = overlay.size
    sx, sy = pw / w, ph / h
    mask_path = within(run_dir, geometry["region_mask_path"], must_exist=True)
    group = plan.get("accident_group")
    anchor = geometry.get("anchor_point")
    mode = "EFFECTIVE INPUT" if effective else "REQUESTED / NOT EXECUTED"

    def point(value):
        return "(" + ", ".join(f"{x:.2f}" for x in value) + ")"

    lines = [
        f"{cid} | {plan['operation']} | seed={plan['seed']}",
        f"{mode} | normalized original px | {w} x {h}",
        f"S start={point(geometry['source_point'])} | E end={point(geometry['target_point'])}",
        f"area={area} px ({area / (w * h):.2%}) | blue=selected area | arrow=intended movement",
        f"CLIP physical: {record['clip']['physical_pass_count']}/4 | accident #5: PASS | accepted",
        "White outline=source segment | range=edit area, not displacement",
    ]
    if group:
        lines.extend(
            [
                f"WHOLE ACCIDENT GROUP: {len(group['member_ids'])} vehicles | one shared translation",
                "Green outline=PLANNED destination, not measured vehicle locations",
                "CLIP acceptance does not establish group movement success; inspect motion diagnostics.",
            ]
        )
        for member in record.get("generation_info", {}).get("motion_diagnostics", {}).get("members", []):
            shift = member.get("estimated_displacement_pixels")
            lines.append(
                f"Motion diagnostic {member['segment_id']}: estimated shift={shift}, "
                f"residual={member.get('residual_pixels')} px | {member['status']}"
            )
    for pair in record["clip"]["pairs"]:
        lines.append(
            f"#{pair['id']} A={pair['undesired_score']:.4f} B={pair['preferred_score']:.4f} "
            f"margin={pair['margin']:+.4f} {'PASS' if pair['passed'] else 'FAIL'}"
        )
    timing = record.get("timing", {})
    lines.append(
        f"Attempt seconds: {timing.get('attempt_elapsed_seconds', 0):.2f} | "
        f"Including failures: {timing.get('accepted_cost_seconds', 0):.2f}"
    )
    if anchor:
        lines.append(f"A anchor={point(anchor)}")
    if effective:
        lines.append(
            f"Requested S={point(plan['source_point'])} E={point(plan['target_point'])}; backend grid/rounding recorded"
        )
    if plan["operation_params"]:
        lines.append(params_summary(plan["operation_params"]))
    font = _font()
    artifacts = {}

    def panel_image(panels, headers):
        total_width = max(640, len(panels) * pw + (len(panels) + 1) * 16)
        wrapped = [
            part for line in lines for part in textwrap.wrap(line, width=max(30, (total_width - 32) // 9))
        ]
        canvas = Image.new("RGB", (total_width, ph + 64 + 24 * len(wrapped)), "#171B24")
        d = ImageDraw.Draw(canvas)
        for i, (panel, heading) in enumerate(zip(panels, headers)):
            canvas.paste(panel.convert("RGB"), (16 + i * (pw + 16), 36))
            d.text((16 + i * (pw + 16), 10), heading, fill="white", font=font)
        for i, line in enumerate(wrapped):
            d.text((16, ph + 48 + 24 * i), line, fill="#EEEEEE", font=font)
        return canvas

    if settings["save_drag_plan"]:
        atomic_image(
            within(run_dir, destinations["drag_plan"]),
            panel_image([overlay], ["Original + drag settings (S / E / A)"]),
        )
        artifacts["drag_plan"] = destinations["drag_plan"]
    candidate_path = record.get("image_path")
    suppress = record.get("delete_image", False)
    if settings["save_comparison"] and candidate_path and not suppress:
        candidate = load_rgb(within(run_dir, candidate_path, must_exist=True))
        candidate.thumbnail((pw, ph), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (pw, ph), "#171B24")
        canvas.paste(candidate, ((pw - candidate.width) // 2, (ph - candidate.height) // 2))
        atomic_image(
            within(run_dir, destinations["comparison"]),
            panel_image([overlay, canvas], ["Original + drag settings", "Generated image"]),
        )
        artifacts["comparison"] = destinations["comparison"]
    metadata = {
        "source_id": source["source_id"],
        "candidate_id": cid,
        "plan_hash": digest(plan),
        "source_pixel_sha256": source["pixel_sha256"],
        "source_preprocessing": source.get("preprocessing"),
        "candidate_pixel_sha256": record.get("pixel_sha256"),
        "original_size": [w, h],
        "requested": plan,
        "effective": effective,
        "display_mode": mode,
        "display_geometry": geometry,
        "mask_sha256": file_hash(mask_path),
        "area_pixels": area,
        "area_fraction": area / (w * h),
        "area_coordinate_system": "normalized_original",
        "preview_transform": {"scale": [sx, sy], "offset": [16, 36]},
        "clip": record["clip"],
        "timing": record.get("timing"),
        "segment_id": plan.get("segment_id"),
        "accident_group": group,
        "motion_diagnostics": record.get("generation_info", {}).get("motion_diagnostics"),
        "candidate_status": record["status"],
        "renderer_version": RENDERER_VERSION,
        "layout_version": LAYOUT_VERSION,
        "settings_hash": digest(settings),
        "artifacts": artifacts,
        "status": "ready",
    }
    if settings["save_metadata"]:
        artifacts["metadata"] = destinations["metadata"]
        atomic_json(within(run_dir, destinations["metadata"]), metadata)
    return {
        "status": "ready",
        "artifacts": artifacts,
        "renderer_version": RENDERER_VERSION,
        "layout_version": LAYOUT_VERSION,
    }
