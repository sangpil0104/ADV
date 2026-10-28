"""Explicit downloads only; v2 workers are always offline."""

from __future__ import annotations

import subprocess
import urllib.request
from pathlib import Path

import yaml

from .environment import require_advv
from .errors import ConfigError
from .storage import atomic_bytes, atomic_json, file_hash, read_json
from .ground import GROUND_MODEL

FLUX = "black-forest-labs/FLUX.1-dev"
CLIP = "openai/clip-vit-large-patch14"
SEGMENT_URL = "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
SEGMENT_PREFIX = "bf2d0c1e"
SEGMENT_SHA256 = "bf2d0c1efbc936eeee2bc95a48e80ebc86b891f61b0106485937fc29f9315fc0"
PATTERNS = {
    FLUX: [
        "*.json",
        "ae.safetensors",
        "transformer/*.safetensors",
        "text_encoder/*.safetensors",
        "text_encoder_2/*.safetensors",
        "vae/*.safetensors",
        "tokenizer/*",
        "tokenizer_2/*",
    ],
    "Tencent/InstantCharacter": ["instantcharacter_ip-adapter.bin"],
    "google/siglip-so400m-patch14-384": ["*.json", "*.safetensors", "*.model", "*.txt"],
    "facebook/dinov2-giant": ["*.json", "*.safetensors"],
    CLIP: ["*.json", "*.safetensors", "*.txt"],
    GROUND_MODEL: ["config.json", "preprocessor_config.json", "pytorch_model.bin"],
}


def prepare(project: Path, *, only: str = "all") -> dict:
    require_advv()
    from huggingface_hub import snapshot_download

    project = project.resolve()
    lock = read_json(project / "configs/upstream.lock.json")
    destination = project / "models/weights.clip.lock.json"
    saved = {"schema_version": "2.0", "models": {}}
    if destination.exists():
        saved = read_json(destination)
    for repo in PATTERNS:
        revision = lock["models"][repo]
        # Reuse local snapshots under the current repository path after a directory rename.
        folder = project / "models/hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
        wanted = (
            only == "all"
            or only == "clip"
            and repo == CLIP
            or only == "dragflow"
            and repo != GROUND_MODEL
            or only in ("segmenter", "ground")
            and repo == GROUND_MODEL
            or only == "support"
            and repo != FLUX
        )
        missing = not folder.is_dir() or (
            repo == GROUND_MODEL and not all((folder / name).is_file() for name in PATTERNS[repo])
        )
        if missing and wanted:
            print(f"Preparing {repo}@{revision}", flush=True)
            folder = Path(
                snapshot_download(
                    repo,
                    revision=revision,
                    allow_patterns=PATTERNS[repo],
                    cache_dir=str(project / "models/hub"),
                    max_workers=4,
                )
            )
        if folder.is_dir():
            saved["models"][repo] = {"revision": revision, "path": str(folder)}
            if repo == GROUND_MODEL and all((folder / name).is_file() for name in PATTERNS[repo]):
                saved["models"][repo]["files_sha256"] = {
                    name: file_hash(folder / name) for name in PATTERNS[repo]
                }
    weights_path = project / "models/segmentation" / SEGMENT_URL.rsplit("/", 1)[1]
    if not weights_path.exists() and only in ("all", "segmenter"):
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        temp = weights_path.with_suffix(".download")
        try:
            with urllib.request.urlopen(SEGMENT_URL, timeout=60) as response, temp.open("wb") as output:
                import shutil

                shutil.copyfileobj(response, output)
            if file_hash(temp) != SEGMENT_SHA256:
                raise ConfigError("Official segmentation checkpoint SHA256 mismatch")
            temp.replace(weights_path)
        finally:
            temp.unlink(missing_ok=True)
    if weights_path.is_file():
        checksum = file_hash(weights_path)
        if checksum != SEGMENT_SHA256:
            raise ConfigError("Segmentation checkpoint differs from the pinned official weights")
        saved["segmentation"] = {"url": SEGMENT_URL, "path": str(weights_path), "sha256": checksum}
    atomic_json(destination, saved)
    if (
        set(saved["models"]) >= set(PATTERNS)
        and "segmentation" in saved
        and "files_sha256" in saved["models"][GROUND_MODEL]
    ):
        write_local_config(project, saved, lock)
    return saved


def write_local_config(project: Path, weights: dict, lock: dict) -> Path:
    path = project / "configs/advv.clip.local.yaml"
    example = yaml.safe_load((project / "configs/advv.example.yaml").read_text())
    if path.exists():
        cfg = yaml.safe_load(path.read_text())
        # Upgrade only the new ground fields; keep the user's generator/prompt/GPU settings.
        cfg.setdefault("ground", example["ground"])
        if cfg["sampler"]["version"] == "accident_group_v1":
            cfg["sampler"]["version"] = example["sampler"]["version"]
    else:
        cfg = example
    cfg["generator"].update(revision=lock["dragflow_commit"], weights_lock="../models/weights.clip.lock.json")
    cfg["verifier"].update(model_path=weights["models"][CLIP]["path"], model_revision=lock["models"][CLIP])
    cfg["segmentation"].update(
        weights_path=weights["segmentation"]["path"], weights_sha256=weights["segmentation"]["sha256"]
    )
    ground = weights["models"][GROUND_MODEL]
    cfg["ground"].update(
        model_path=ground["path"], model_revision=ground["revision"], files_sha256=ground["files_sha256"]
    )
    atomic_bytes(path, yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True).encode())
    return path


def git_revision(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def verify_weights(cfg: dict) -> dict:
    path = cfg["generator"].get("weights_lock")
    if not path or not Path(path).is_file():
        raise ConfigError("Missing v2 weights lock; run advv prepare")
    weights = read_json(Path(path))
    for repo in PATTERNS:
        entry = weights.get("models", {}).get(repo)
        if not entry or not Path(entry["path"]).is_dir() or len(entry["revision"]) != 40:
            raise ConfigError(f"Missing pinned local weights: {repo}")
    clip = weights["models"][CLIP]
    v = cfg["verifier"]
    if not v.get("model_path") or (
        Path(v["model_path"]).resolve() != Path(clip["path"]).resolve()
        or v["model_revision"] != clip["revision"]
    ):
        raise ConfigError("CLIP config and weights lock disagree")
    seg = cfg["segmentation"]
    checkpoint = Path(seg["weights_path"])
    if (
        not checkpoint.is_file()
        or not seg.get("weights_sha256")
        or file_hash(checkpoint) != seg["weights_sha256"]
        or seg["weights_sha256"] != SEGMENT_SHA256
    ):
        raise ConfigError("Missing/changed local segmentation checkpoint; run advv prepare")
    saved_seg = weights.get("segmentation", {})
    if saved_seg.get("sha256") != seg["weights_sha256"]:
        raise ConfigError("Segmentation config and weights lock disagree")
    entry, ground = weights["models"][GROUND_MODEL], cfg["ground"]
    if (
        not ground.get("model_path")
        or Path(ground["model_path"]).resolve() != Path(entry["path"]).resolve()
        or ground["model_revision"] != entry["revision"]
        or ground["files_sha256"] != entry.get("files_sha256")
        or set(ground["files_sha256"]) != set(PATTERNS[GROUND_MODEL])
    ):
        raise ConfigError("Ground config and weights lock disagree; run advv prepare --only ground")
    for name, checksum in ground["files_sha256"].items():
        asset = Path(ground["model_path"]) / name
        if not asset.is_file() or file_hash(asset) != checksum:
            raise ConfigError("Missing/changed ground checkpoint; run advv prepare --only ground")
    return weights
