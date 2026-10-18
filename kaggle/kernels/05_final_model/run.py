"""Kaggle kernel: paddy-05-final-model

Fits and saves the final random forests on all published labels (+ split-conformal q).
Thin wrapper; logic in src/experiments/final_model.py. Outputs: /kaggle/working/05_final_model/.
"""

import sys
from pathlib import Path

CODE_SLUG = "paddy-yield-code"     # keep in sync with configs/config.yaml -> kaggle.code_slug
DATA_SLUG = "paddy-master-table"   # keep in sync with configs/config.yaml -> kaggle.data_slug
EXPERIMENT = "05_final_model"

import os
import zipfile

INPUT = Path("/kaggle/input")


def find_all(name: str) -> list[Path]:
    """os.walk with followlinks: Path.rglob skips symlinked dataset mounts on newer images."""
    return [Path(r) / name for r, _, files in os.walk(INPUT, followlinks=True) if name in files]


def show_input_tree(max_lines: int = 60) -> None:
    print("contents of /kaggle/input:")
    n = 0
    for r, dirs, files in os.walk(INPUT, followlinks=True):
        depth = Path(r).relative_to(INPUT).parts
        if len(depth) > 4:
            dirs[:] = []
            continue
        print("  " * len(depth) + (Path(r).name or "/kaggle/input") + "/", files[:6])
        n += 1
        if n >= max_lines:
            print("  ...")
            return


show_input_tree()
# code dataset: a folder holding COMMIT + src/ (auto-extracted zip), or the zip itself
candidates = [p for p in find_all("COMMIT") if (p.parent / "src").is_dir()]
if not candidates:
    for z in find_all("repo.zip"):
        dest = Path("/kaggle/working/_code")
        zipfile.ZipFile(z).extractall(dest)
        candidates = [Path(r) / "COMMIT" for r, _, f in os.walk(dest) if "COMMIT" in f]
        break
if not candidates:
    raise SystemExit("Code dataset not found under /kaggle/input (see the listing above). "
                     "Check dataset_sources in kernel-metadata.json and that the dataset is ready.")
repo = candidates[0].parent
sys.path.insert(0, str(repo))
commit = (repo / "COMMIT").read_text().strip()

from src.config import load_config, set_seed  # noqa: E402
from src.kaggle_env import banner, output_dir  # noqa: E402

cfg = load_config(repo / "configs" / "config.yaml")
set_seed(cfg["project"]["seed"])
data_dir = find_all("master_table.parquet")[0].parent
info = banner(DATA_SLUG, code_commit=commit, data_dir=data_dir)
out = output_dir(EXPERIMENT)
(out / "run_info.json").write_text(__import__("json").dumps(info, indent=2))

print("data dir:", data_dir)
from src.experiments.final_model import main  # noqa: E402

main(data_dir, out, commit=commit)
