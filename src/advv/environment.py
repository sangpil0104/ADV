"""The coordinator and every worker must use the same named conda environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .errors import ConfigError


def require_advv() -> dict:
    prefix = Path(sys.prefix).resolve()
    declared = os.environ.get("CONDA_PREFIX")
    if (
        prefix.name != "ADVV"
        or not (prefix / "conda-meta").is_dir()
        or (declared and Path(declared).resolve() != prefix)
        or sys.prefix != sys.base_prefix
    ):
        raise ConfigError("Use conda activate ADVV or conda run -n ADVV; venv/base are unsupported")
    return {"python": sys.executable, "prefix": str(prefix), "environment": "ADVV"}
