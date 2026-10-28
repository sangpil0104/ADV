from __future__ import annotations

import copy
import math
from pathlib import Path

import yaml

from .errors import ConfigError
from .storage import digest, read_json
from .verifiers.decision import validate_criteria
from .ground import GROUND_MODEL, GROUND_REVISION, GROUND_LABELS

PATH_FIELDS = {
    "run": ["output_root"],
    "dataset": ["root", "input_dir", "manifest"],
    "generator": ["repo_path", "upstream_config", "weights_lock"],
    "verifier": ["model_path", "criteria_path"],
    "segmentation": ["weights_path"],
    "accident_selection": ["captions_path"],
    "ground": ["model_path"],
}
DEFAULT_SOURCE_PROMPT = "A dashcam image of a road traffic accident involving one or more vehicles."
DEFAULT_AUGMENTATION_PROMPT = (
    "Translate all selected accident vehicles together, preserving their relative positions and contact. "
    "Keep visible accident damage and do not turn the scene into normal traffic."
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/advv.clip.local.yaml"


def load_config(
    path: Path, *, input_dir=None, target_count=None, gpus=None, prompt=None, image=None, repair_mode=None
) -> dict:
    path = path.resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("schema_version") != "2.0":
        raise ConfigError("CLIP requires schema 2.0; create a new config/run instead of resuming VQA")
    for section, fields in PATH_FIELDS.items():
        if not isinstance(cfg.get(section), dict):
            raise ConfigError(f"Missing config section: {section}")
        for key in fields:
            if cfg[section].get(key) is not None:
                cfg[section][key] = str((path.parent / Path(cfg[section][key]).expanduser()).resolve())
    if input_dir is not None:
        cfg["dataset"].update(
            input_dir=str(Path(input_dir).resolve()), root=str(Path(input_dir).resolve()), manifest=None
        )
    if target_count is not None:
        cfg["run"]["target_count"] = target_count
    if gpus is not None:
        cfg["execution"]["selected_gpu_ids"] = parse_gpus(gpus)
    if prompt is not None:
        cfg["generation"]["augmentation_prompt"] = prompt
    if image is not None:
        cfg["dataset"]["image"] = str(image)
    if repair_mode is not None:
        cfg["generation"]["repair_mode"] = repair_mode
    cfg["generation"].setdefault("repair_mode", "endpoints")
    cfg["generation"]["source_prompt"] = cfg["generation"].get("source_prompt") or DEFAULT_SOURCE_PROMPT
    raw = cfg["generation"].get("augmentation_prompt")
    if raw is not None and (not isinstance(raw, str) or not raw.strip()):
        raise ConfigError("--prompt / generation.augmentation_prompt must be nonempty text or omitted")
    cfg["generation"]["effective_augmentation_prompt"] = raw or DEFAULT_AUGMENTATION_PROMPT
    cfg["_config_path"] = str(path)
    cfg["_assets"] = {"clip_criteria": read_json(Path(cfg["verifier"]["criteria_path"]))}
    cfg["_assets"]["criteria_sha256"] = digest(cfg["_assets"]["clip_criteria"])
    cfg["_assets"]["accident_selection"] = read_json(Path(cfg["accident_selection"]["captions_path"]))
    cfg["_assets"]["accident_selection_sha256"] = digest(cfg["_assets"]["accident_selection"])
    validate_config(cfg)
    if cfg["dataset"].get("image") is not None:
        from .ingest import selected_image_path

        selected_image_path(cfg["dataset"])  # Fail before run cleanup or model preflight.
    return cfg


def parse_gpus(value) -> list[str]:
    values = value.split(",") if isinstance(value, str) else value
    if not isinstance(values, (list, tuple)):
        raise ConfigError("GPU IDs must be a nonempty list")
    result = []
    for item in values:
        item = str(item).strip()
        if not item.isdigit():
            raise ConfigError("GPU IDs must be comma-separated nonnegative integers")
        result.append(str(int(item)))
    if not result or len(set(result)) != len(result):
        raise ConfigError("GPU IDs must be nonempty and unique")
    return result


def gpu_groups(value) -> list[list[str]]:
    """Each independent candidate lane owns exactly two physical GPUs."""
    ids = parse_gpus(value)
    if len(ids) < 2 or len(ids) % 2:
        raise ConfigError("Specify an even number of GPU IDs (at least two), e.g. 4,5,6,7")
    return [ids[i : i + 2] for i in range(0, len(ids), 2)]


def validate_config(cfg: dict) -> None:
    def require(condition, message):
        if not condition:
            raise ConfigError(message)

    require(cfg.get("schema_version") == "2.0", "Legacy VQA runs cannot be resumed by CLIP")
    try:
        r, d, e = cfg["run"], cfg["dataset"], cfg["execution"]
        require(
            cfg["generation"].get("repair_mode", "swept") in ("endpoints", "swept"),
            "generation.repair_mode must be endpoints or swept",
        )
        output_root, dataset_root = Path(r["output_root"]).resolve(), Path(d["root"]).resolve()
        require(
            not output_root.is_relative_to(dataset_root) and not dataset_root.is_relative_to(output_root),
            "Input and output trees must not overlap; cleanup must never contain source inputs",
        )
        require(r["quota_scope"] == "total" and r["strategy"] == "random_until_quota", "Total N required")
        require(r["max_attempts"] is None and r["max_wall_time_seconds"] is None, "No total caps allowed")
        require(r["limit_policy"] == "until_target_or_user_stop", "Invalid limit policy")
        require(r["source_schedule"] == "round_robin" and not r["chain_augmentation"], "Invalid schedule")
        require(type(r["seed"]) is int and 0 <= r["seed"] < 2**32, "Invalid seed")
        require(
            r["target_count"] is None or type(r["target_count"]) is int and r["target_count"] > 0,
            "N must be positive",
        )
        require(d["task"] == "image_only", "Only image-only export is supported")
        require(bool(d.get("input_dir")) != bool(d.get("manifest")), "Choose input_dir or manifest")
        selection = d.get("image")
        if selection is not None:
            require(bool(d.get("input_dir")), "--image requires folder input, not a source manifest")
            require(isinstance(selection, str) and bool(selection.strip()), "--image must be nonempty")
            require(
                not Path(selection).is_absolute() and ".." not in Path(selection).parts,
                "--image must be relative to the input folder without '..' (e.g. image1.png)",
            )
        require(
            bool(d["eligible_splits"]) and set(d["eligible_splits"]) <= {"source", "train"},
            "Cannot augment val/test",
        )
        require(
            d["enforce_group_disjoint"] and d["enforce_cross_split_hash_disjoint"], "Split checks required"
        )
        require(
            e["candidate_concurrency"] == "auto" and e["conda_environment"] == "ADVV",
            "Automatic GPU-pair concurrency in conda ADVV required",
        )
        require(not any(k.endswith("_python") for k in e), "Worker Python overrides/venvs are unsupported")
        require(e["mode"] == "gpu_pairs_parallel", "Invalid execution mode")
        require(
            e["retry_oom"] is False
            and type(e["max_technical_retries"]) is int
            and e["max_technical_retries"] in (0, 1),
            "Invalid retry policy",
        )
        require(
            type(e["worker_timeout_seconds"]) in (int, float)
            and math.isfinite(e["worker_timeout_seconds"])
            and e["worker_timeout_seconds"] > 0,
            "Invalid worker timeout",
        )
        require(
            type(e["max_consecutive_technical_failures"]) is int
            and e["max_consecutive_technical_failures"] > 0,
            "Invalid failure limit",
        )
        if e.get("selected_gpu_ids") is not None:
            e["selected_gpu_ids"] = parse_gpus(e["selected_gpu_ids"])
            gpu_groups(e["selected_gpu_ids"])
        s = cfg["sampler"]
        require(
            s["version"] == "accident_group_ground_v1", "Ground-constrained sampler required; start a new run"
        )
        require(
            s["operations"] == ["relocation"],
            "v2 baseline supports relocation; other mappings need validation",
        )
        require(
            type(s["geometry_attempts"]) is int and 1 <= s["geometry_attempts"] <= 512,
            "Invalid geometry attempts",
        )
        require(
            not {"region_area_fraction", "expansion_pixels"} & s.keys(),
            "Partial vehicle masks are unsupported",
        )
        selector = cfg["accident_selection"]
        require(selector["method"] == "clip_contact_context_v1", "Unsupported accident selector")
        require(
            type(selector["context_padding"]) in (int, float) and 0 <= selector["context_padding"] <= 1,
            "Invalid vehicle context padding",
        )
        asset = cfg["_assets"]["accident_selection"]
        require(
            asset["version"] == selector["method"]
            and asset["rule"] == "both_views_involved_greater_than_uninvolved",
            "Unsupported accident selection rule",
        )
        require(
            set(asset["criteria"]) == {"single", "contact_group"}
            and all(
                isinstance(asset["criteria"][v][k], str) and asset["criteria"][v][k].strip()
                for v in asset["criteria"]
                for k in ("uninvolved", "involved")
            ),
            "Invalid selection captions",
        )
        require(
            digest(asset) == cfg["_assets"]["accident_selection_sha256"], "Selection caption hash mismatch"
        )
        v, seg = cfg["verifier"], cfg["segmentation"]
        require(
            v["backend"] == "clip_local" and cfg["generator"]["backend"] == "dragflow",
            "Real DragFlow and CLIP required",
        )
        require(
            v["local_files_only"] is True and v["preprocessing"] == "letterbox_bicubic_gray_v1",
            "Pinned local whole-frame CLIP required",
        )
        require(
            v["model_id"] == "openai/clip-vit-large-patch14"
            and isinstance(v["model_revision"], str)
            and len(v["model_revision"]) == 40,
            "Pinned CLIP revision required",
        )
        require(
            seg["backend"] == "torchvision_maskrcnn_resnet50_fpn" and seg["weights_id"] == "COCO_V1",
            "Unsupported segmenter",
        )
        require(seg["class_ids"] == [3, 4, 6, 8], "Vehicle classes are car, motorcycle, bus, truck")
        for name in ("score_threshold", "mask_threshold"):
            require(
                type(seg[name]) in (int, float) and math.isfinite(seg[name]) and 0 < seg[name] <= 1,
                f"Invalid segmentation {name}",
            )
        require(type(seg["min_area_pixels"]) is int and seg["min_area_pixels"] > 0, "Invalid segment area")
        ground = cfg["ground"]
        require(ground["backend"] == "segformer_cityscapes", "Unsupported ground segmenter")
        require(
            ground["model_id"] == GROUND_MODEL and ground["model_revision"] == GROUND_REVISION,
            "Pinned ground model revision required",
        )
        require(ground["local_files_only"] is True, "Ground inference must be offline")
        require(
            isinstance(ground["class_ids"], list)
            and bool(ground["class_ids"])
            and all(type(i) is int and i in GROUND_LABELS for i in ground["class_ids"])
            and len(set(ground["class_ids"])) == len(ground["class_ids"]),
            "Invalid ground classes",
        )
        require(
            type(ground["confidence_threshold"]) in (int, float) and 0 < ground["confidence_threshold"] <= 1,
            "Invalid ground confidence",
        )
        require(
            type(ground["contact_points_per_vehicle"]) is int
            and 2 <= ground["contact_points_per_vehicle"] <= 16,
            "Invalid ground contact point count",
        )
        require(
            type(ground["contact_band_fraction"]) in (int, float)
            and 0 < ground["contact_band_fraction"] <= 0.25,
            "Invalid ground contact band",
        )
        require(isinstance(ground["files_sha256"], dict), "Ground checkpoint hashes required")
        raw = cfg["generation"].get("augmentation_prompt")
        require(raw is None or isinstance(raw, str) and bool(raw.strip()), "Empty augmentation prompt")
        require(
            isinstance(cfg["generation"]["source_prompt"], str)
            and bool(cfg["generation"]["source_prompt"].strip()),
            "Empty source prompt",
        )
        require(
            cfg["generation"]["effective_augmentation_prompt"] == (raw or DEFAULT_AUGMENTATION_PROMPT),
            "Effective generation prompt mismatch",
        )
        require(
            cfg["storage"]
            == {"atomic_writes": True, "retain_rejected_details": False, "cleanup_existing": True},
            "Nonaccepted details must be removed",
        )
        require(
            cfg["export"]["reject_fake_backend"]
            and cfg["export"]["deduplicate_exact_pixels"]
            and cfg["export"]["deduplicate_against_source"],
            "Export integrity required",
        )
        vis = cfg["visualization"]
        require(
            vis["enabled"] and vis["save_drag_plan"] and vis["save_comparison"] and vis["save_metadata"],
            "Accepted visualizations are required",
        )
        require(
            0 <= vis["area_alpha"] <= 1
            and type(vis["max_panel_side_px"]) is int
            and vis["max_panel_side_px"] >= 64,
            "Invalid visualization settings",
        )
        validate_criteria(cfg["_assets"]["clip_criteria"])
        require(
            digest(cfg["_assets"]["clip_criteria"]) == cfg["_assets"]["criteria_sha256"],
            "CLIP criteria hash mismatch",
        )
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"Invalid or missing config field: {exc}") from exc


def recipe_hash(cfg: dict) -> str:
    value = copy.deepcopy(cfg)
    value.pop("_config_path", None)
    value["execution"].pop("selected_gpu_ids", None)
    return digest(value)
