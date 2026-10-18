"""One-command entry point: `python -m src.run_all`.

Default (local, minutes): rebuild everything that does NOT need model training, from the raw
files and the saved Earth Engine extracts, then all report tables/figures from saved predictions.

  1. DCS yields -> tidy table + sanity checks        (src.data.build_yield)
  2. master table, lead 0                            (src.data.build_master)
  3. lead / window-shift variant tables              (src.data.build_master --lead/--shift)
  4. descriptive figures                             (src.analysis.eda_plots)
  5. shock-season analysis                           (src.analysis.shocks)
  6. report figures + results/predictions.csv        (src.analysis.report_figures)

Model training runs on Kaggle (kernels 01-05; see kaggle/README.md). The same code can run
locally with --experiments (slow: about 6-8 hours on a laptop CPU):
  python -m src.run_all --experiments
Earth Engine extraction (src.data.ee_extract) is a separate, credentialed step and is never
re-run automatically.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.config import REPO_ROOT, load_config, set_seed


def _master_variants() -> None:
    from src.data.build_master import main as build_master
    cfg = load_config()
    for lead in cfg["earliness"]["lead_weeks_before_harvest"]:
        if lead:
            build_master(lead, 0)
    for shift in cfg["features"]["window_shift_sensitivity_weeks"]:
        build_master(0, shift)


def local_stages() -> list[tuple[str, object]]:
    from src.analysis import eda_plots, report_figures, shocks
    from src.data import build_master, build_yield
    return [
        ("1 tidy DCS yields", build_yield.main),
        ("2 master table (lead 0)", lambda: build_master.main(0, 0)),
        ("3 lead / shift variant tables", _master_variants),
        ("4 descriptive figures", eda_plots.main),
        ("5 shock-season analysis",
         lambda: shocks.main(REPO_ROOT / "results" / "tables" / "predictions_models.csv")),
        ("6 report figures + predictions.csv", report_figures.main),
    ]


def experiment_stages() -> list[tuple[str, object]]:
    from src.experiments import ablation, conformal, earliness, final_model, models
    inp, runs = REPO_ROOT / "data" / "processed", REPO_ROOT / "results" / "runs"
    return [
        ("E1 models + Wilcoxon + k-fold demo", lambda: models.main(inp, runs / "01_models")),
        ("E2 conformal intervals", lambda: conformal.main(inp, runs / "02_conformal")),
        ("E3 earliness + window shift", lambda: earliness.main(inp, runs / "03_earliness")),
        ("E4 ablation + SHAP", lambda: ablation.main(inp, runs / "04_ablation_shap")),
        ("E5 final models", lambda: final_model.main(inp, runs / "05_final_model")),
    ]


def main(experiments: bool = False) -> None:
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    stages = local_stages() + (experiment_stages() if experiments else [])
    for name, fn in stages:
        t0 = time.time()
        print(f"== {name}", flush=True)
        fn()
        print(f"   done in {time.time() - t0:.0f}s", flush=True)
    if not experiments:
        print("Training stages were not run (they run on Kaggle). Saved results are in "
              f"{Path('results') / 'tables'}; add --experiments to retrain locally.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", action="store_true",
                    help="also run the training experiments locally (slow)")
    main(ap.parse_args().experiments)
