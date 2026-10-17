"""Environment check: python version, platform (local Windows vs Kaggle), packages.

Run: python scripts/env_check.py
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys

PACKAGES = {
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "scikit-learn": "sklearn",
    "lightgbm": "lightgbm", "xgboost": "xgboost", "shap": "shap", "statsmodels": "statsmodels",
    "earthengine-api": "ee", "geemap": "geemap", "geopandas": "geopandas",
    "rasterio": "rasterio", "mlflow": "mlflow", "pytest": "pytest", "ruff": "ruff",
    "matplotlib": "matplotlib", "seaborn": "seaborn", "pyarrow": "pyarrow", "pyyaml": "yaml",
}


def where_am_i() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.exists("/kaggle"):
        return "kaggle"
    if platform.system() == "Windows":
        return "windows-local"
    return f"{platform.system().lower()}-other"


def main() -> int:
    print(f"environment : {where_am_i()}")
    print(f"python      : {sys.version.split()[0]}  (target 3.11/3.12)")
    print(f"executable  : {sys.executable}")
    print(f"in venv     : {sys.prefix != sys.base_prefix}")
    missing = []
    for pip_name, mod in PACKAGES.items():
        ok = importlib.util.find_spec(mod) is not None
        print(f"  [{'ok' if ok else '--'}] {pip_name}")
        if not ok:
            missing.append(pip_name)
    if missing:
        print(f"\nMissing: {', '.join(missing)}\nInstall with: pip install -r requirements.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
