"""Kaggle kernel: paddy-04-ablation-shap

Drop-one-feature-group ablation and SHAP for the best tree model

Thin wrapper. All logic lives in src/ (attached as the private code dataset).
Inputs: /kaggle/input/<data_slug>/ (read-only). Outputs: /kaggle/working/04_ablation_shap/ only.
Re-running resumes: finished units are skipped if their output file already exists.
"""

import sys
from pathlib import Path

CODE_SLUG = "paddy-yield-code"     # keep in sync with configs/config.yaml -> kaggle.code_slug
DATA_SLUG = "paddy-master-table"   # keep in sync with configs/config.yaml -> kaggle.data_slug
EXPERIMENT = "04_ablation_shap"

# Kaggle mounts datasets at /kaggle/input/<slug>/ (newer images may nest under datasets/<user>/)
candidates = list(Path("/kaggle/input").rglob("repo/COMMIT"))
if not candidates:
    raise SystemExit("Code dataset not attached: add it to dataset_sources in kernel-metadata.json")
repo = candidates[0].parent
sys.path.insert(0, str(repo))
commit = (repo / "COMMIT").read_text().strip()

from src.config import load_config, set_seed  # noqa: E402
from src.kaggle_env import banner, output_dir  # noqa: E402

cfg = load_config(repo / "configs" / "config.yaml")
set_seed(cfg["project"]["seed"])
info = banner(DATA_SLUG, code_commit=commit)
out = output_dir(EXPERIMENT)
(out / "run_info.json").write_text(__import__("json").dumps(info, indent=2))

# Wired up when the corresponding task is implemented. Nothing is faked meanwhile.
raise NotImplementedError("Experiment '04_ablation_shap' is not implemented yet.")
