"""Prepare the two PRIVATE Kaggle datasets and the kernel folders.

    python scripts/kaggle_prepare.py data      # master table + manifest -> kaggle/_build/data
    python scripts/kaggle_prepare.py code      # src + configs -> kaggle/_build/code
    python scripts/kaggle_prepare.py kernels   # fill <USERNAME> into kernel-metadata.json copies

It only writes files. Uploading is done by you with the Kaggle CLI (see kaggle/README.md),
so your token never passes through this script.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config  # noqa: E402

BUILD = ROOT / "kaggle" / "_build"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        return bool(out.strip())
    except Exception:
        return True


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _username(cfg: dict) -> str:
    user = cfg["kaggle"]["username"]
    if not user or user == "TBD":
        sys.exit("Set kaggle.username in configs/config.yaml first.")
    return user


def prepare_data(cfg: dict) -> None:
    src = ROOT / cfg["paths"]["master_table"]
    if not src.exists():
        sys.exit(f"{src} not found: build the master table (Task 3) first.")
    out = BUILD / "data"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    shutil.copy2(src, out / src.name)
    # everything later experiments need to rebuild features at other lead times / shifts
    extras = ["data/processed/feature_list.json", "data/processed/series_rain_daily.parquet",
              "data/processed/series_modis.parquet", "data/processed/series_era5_daily.parquet",
              "data/interim/dcs_yield_tidy.csv", "docs/data_dictionary.md", "docs/data_card.md"]
    for extra in extras:
        if (ROOT / extra).exists():
            shutil.copy2(ROOT / extra, out / Path(extra).name)
    manifest = {
        "file": src.name,
        "sha256": _sha256(src),
        "files": {p.name: _sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
        "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    user = _username(cfg)
    meta = {
        "title": cfg["kaggle"]["data_title"],
        "id": f"{user}/{cfg['kaggle']['data_slug']}",
        "licenses": [{"name": "other"}],
        "description": "PRIVATE. Processed district-season master table for the paddy "
                       "yield forecasting project. Not for redistribution.",
    }
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if manifest["git_dirty"]:
        print("WARNING: uncommitted changes; commit first so the manifest commit is exact.")


def prepare_code(cfg: dict) -> None:
    out = BUILD / "code"
    shutil.rmtree(out, ignore_errors=True)
    (out / "repo").mkdir(parents=True)
    for item in ["src", "configs", "requirements.txt", "pyproject.toml"]:
        p = ROOT / item
        if p.is_dir():
            shutil.copytree(p, out / "repo" / item,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(p, out / "repo" / item)
    (out / "repo" / "COMMIT").write_text(_git_commit(), encoding="utf-8")
    user = _username(cfg)
    meta = {
        "title": cfg["kaggle"]["code_title"],
        "id": f"{user}/{cfg['kaggle']['code_slug']}",
        "licenses": [{"name": "other"}],
        "description": "PRIVATE. Source code snapshot (src/, configs/) for the paddy kernels.",
    }
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"code bundle at {out} (commit {_git_commit()[:8]}, dirty={_git_dirty()})")


def prepare_kernels(cfg: dict) -> None:
    user = _username(cfg)
    for kdir in sorted((ROOT / "kaggle" / "kernels").iterdir()):
        if not kdir.is_dir():
            continue
        out = BUILD / "kernels" / kdir.name
        shutil.rmtree(out, ignore_errors=True)
        shutil.copytree(kdir, out)
        meta_path = out / "kernel-metadata.json"
        text = meta_path.read_text(encoding="utf-8").replace("<USERNAME>", user)
        meta_path.write_text(text, encoding="utf-8")
        print(f"kernel ready: {out}")


if __name__ == "__main__":
    cfg = load_config()
    actions = {"data": prepare_data, "code": prepare_code, "kernels": prepare_kernels}
    if len(sys.argv) != 2 or sys.argv[1] not in actions:
        sys.exit(f"usage: python scripts/kaggle_prepare.py {{{'|'.join(actions)}}}")
    actions[sys.argv[1]](cfg)
