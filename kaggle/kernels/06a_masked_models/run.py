"""Kaggle kernel: paddy-06a-masked-models  (Step 6, stage 'models' of src.experiments.masked_compare)

Thin wrapper. All logic lives in src/ (attached as the private code dataset paddy-yield-code).
Inputs: /kaggle/input/paddycast-masked/ (read-only). Outputs: /kaggle/working/06a_masked_models/ only.
Re-running resumes: finished units are skipped if their output file already exists.
"""

import json
import os
import sys
import zipfile
from pathlib import Path

STAGE = "models"
EXPERIMENT = "06a_masked_models"
DATA_SLUG = "paddycast-masked"
INPUT = Path("/kaggle/input")


def find_all(name: str) -> list[Path]:
    """os.walk with followlinks: Path.rglob skips symlinked dataset mounts on newer images."""
    return [Path(r) / name for r, _, files in os.walk(INPUT, followlinks=True) if name in files]


for r, _, files in os.walk(INPUT, followlinks=True):
    if len(Path(r).relative_to(INPUT).parts) <= 3:
        print(r, files[:8])

candidates = [p for p in find_all("COMMIT") if (p.parent / "src").is_dir()]
if not candidates:
    for z in find_all("repo.zip"):
        dest = Path("/kaggle/working/_code")
        zipfile.ZipFile(z).extractall(dest)
        candidates = [Path(r) / "COMMIT" for r, _, f in os.walk(dest) if "COMMIT" in f]
        break
if not candidates:
    raise SystemExit("Code dataset not found under /kaggle/input (see the listing above).")
repo = candidates[0].parent
sys.path.insert(0, str(repo))
commit = (repo / "COMMIT").read_text().strip()
if not (repo / "src" / "experiments" / "masked_compare.py").exists():
    raise SystemExit("code dataset is older than Step 6: version paddy-yield-code first")

found = find_all("master_masked.parquet")
if not found:
    raise SystemExit("master_masked.parquet not found: version paddycast-masked with -t "
                     "(keep parquet, no CSV conversion)")
data_dir = found[0].parent
for need in ("master_unmasked_7d.parquet", "feature_list.json"):
    if not (data_dir / need).exists():
        raise SystemExit(f"{need} missing from {data_dir}")

from src.config import load_config, set_seed  # noqa: E402
from src.kaggle_env import banner, output_dir  # noqa: E402

cfg = load_config(repo / "configs" / "config.yaml")
set_seed(cfg["project"]["seed"])
info = banner(DATA_SLUG, code_commit=commit, data_dir=data_dir)
out = output_dir(EXPERIMENT)
(out / "run_info.json").write_text(json.dumps(info, indent=2))
print("data dir:", data_dir, "stage:", STAGE)

from src.experiments import masked_compare as mc  # noqa: E402

{"models": mc.stage_models, "followups": mc.stage_followups}[STAGE](data_dir, out)
