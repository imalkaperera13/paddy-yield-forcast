"""Kaggle kernel: paddy-01-baselines-and-models

Baselines, ridge, RF, LightGBM, XGBoost, hybrid; Wilcoxon vs baseline;
random k-fold vs forward-chaining demo.

Thin wrapper. All logic lives in src/ (attached as the private code dataset).
Inputs: /kaggle/input/<data_slug>/ (read-only). Outputs: /kaggle/working/01_models/ only.
Re-running resumes: finished units are skipped if their output file already exists.
"""

import sys
from pathlib import Path

CODE_SLUG = "paddy-yield-code"     # keep in sync with configs/config.yaml -> kaggle.code_slug
DATA_SLUG = "paddy-master-table"   # keep in sync with configs/config.yaml -> kaggle.data_slug
EXPERIMENT = "01_models"

# Kaggle mounts datasets at /kaggle/input/<slug>/ (newer images may nest under datasets/<user>/)
candidates = list(Path("/kaggle/input").rglob("repo/COMMIT"))
if not candidates:   # code dataset uploaded with --dir-mode zip and not auto-extracted
    import zipfile
    for z in Path("/kaggle/input").rglob("repo.zip"):
        dest = Path("/kaggle/working/_code")
        zipfile.ZipFile(z).extractall(dest)
        candidates = list(dest.rglob("COMMIT"))
        break
if not candidates:
    raise SystemExit("Code dataset not attached: add it to dataset_sources in kernel-metadata.json")
repo = candidates[0].parent
sys.path.insert(0, str(repo))
commit = (repo / "COMMIT").read_text().strip()

from src.config import load_config, set_seed  # noqa: E402
from src.kaggle_env import banner, output_dir  # noqa: E402

cfg = load_config(repo / "configs" / "config.yaml")
set_seed(cfg["project"]["seed"])
data_dir = next(Path("/kaggle/input").rglob("master_table.parquet")).parent
info = banner(DATA_SLUG, code_commit=commit, data_dir=data_dir)
out = output_dir(EXPERIMENT)
(out / "run_info.json").write_text(__import__("json").dumps(info, indent=2))

print("data dir:", data_dir)
from src.experiments.models import main  # noqa: E402

main(data_dir, out)

