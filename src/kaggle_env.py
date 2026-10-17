"""Paths and reproducibility banner shared by local runs and Kaggle kernels.

On Kaggle, inputs are read-only under /kaggle/input/<dataset>/ and every output
must go to /kaggle/working/. Locally the same code uses the repo folders.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import sys
from pathlib import Path

ON_KAGGLE = Path("/kaggle/input").is_dir()

TRACKED_LIBS = ["numpy", "pandas", "scipy", "sklearn", "lightgbm", "xgboost",
                "shap", "statsmodels", "pyarrow", "mlflow", "matplotlib"]


def input_dir(data_slug: str) -> Path:
    """Folder holding master_table.parquet and manifest.json."""
    if ON_KAGGLE:
        return Path("/kaggle/input") / data_slug
    from src.config import REPO_ROOT
    return REPO_ROOT / "data" / "processed"


def output_dir(experiment: str) -> Path:
    """Writable folder for one experiment group's outputs."""
    if ON_KAGGLE:
        base = Path("/kaggle/working")
    else:
        from src.config import REPO_ROOT
        base = REPO_ROOT / "results" / "runs"
    out = base / experiment
    out.mkdir(parents=True, exist_ok=True)
    return out


def _lib_versions() -> dict[str, str]:
    versions = {}
    for name in TRACKED_LIBS:
        try:
            versions[name] = getattr(importlib.import_module(name), "__version__", "?")
        except ImportError:
            versions[name] = "not installed"
    return versions


def banner(data_slug: str, code_commit: str | None = None) -> dict:
    """Print and return python, library, dataset and code versions.

    The dataset version is read from manifest.json, which is written next to the
    master table when it is published (sha256 + build time + git commit).
    """
    manifest_path = input_dir(data_slug) / "manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {"warning": "manifest.json not found"})
    info = {
        "on_kaggle": ON_KAGGLE,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libs": _lib_versions(),
        "dataset_manifest": manifest,
        "code_commit": code_commit or os.environ.get("CODE_COMMIT", "unknown"),
        "kaggle_kernel_run": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "local"),
    }
    print("=" * 60)
    print(json.dumps(info, indent=2))
    print("=" * 60)
    return info
