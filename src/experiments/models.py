"""Experiment group 1: baselines + models, Wilcoxon, diagnostics, sensitivity, k-fold demo.

    python -m src.experiments.models --input data/processed --out results/runs/01_models
On Kaggle this is called by kaggle/kernels/01_models/run.py. Every step caches its output,
so re-running after a stopped session continues where it left off.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from src.analysis.stats import (
    common_rows,
    comparison_table,
    diagnostics,
    kfold_demo,
    per_fold_table,
    to_markdown,
    wilcoxon_vs_baseline,
)
from src.config import load_config, set_seed
from src.models.train import run_model

MODEL_ORDER = ["baseline_last5", "ridge", "random_forest", "lightgbm", "xgboost", "hybrid_lgbm"]
LDO_FIXED = {"baseline_last5": None, "ridge": {"alpha": 10.0},
             "lightgbm": {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 7}}


def load_inputs(input_dir: Path, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(input_dir / "master_table.parquet")
    fl = json.loads((input_dir / "feature_list.json").read_text(encoding="utf-8"))
    features = [c for g in fl["groups"].values() for c in g]
    bad = set(features) & set(cfg["forbidden_features"])
    if bad:
        raise RuntimeError(f"forbidden features in feature list: {bad}")
    return df, features


def fmt_pm(t: pd.DataFrame, cols: list[str], digits: dict) -> pd.DataFrame:
    out = t[["model"]].copy()
    for c in cols:
        d = digits.get(c, 1)
        out[c] = [f"{m:.{d}f} ± {s:.{d}f}" if pd.notna(s) else f"{m:.{d}f}"
                  for m, s in zip(t[f"{c}_mean"], t[f"{c}_std"], strict=True)]
    return out


def main(input_dir: Path, out_dir: Path, seeds: list[int] | None = None,
         models: list[str] | None = None, skip_extra: bool = False) -> None:
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    env_s, env_m = os.environ.get("PADDY_SEEDS"), os.environ.get("PADDY_MODELS")
    seeds = seeds or ([int(x) for x in env_s.split(",")] if env_s else
                      cfg["project"]["seeds_for_spread"])
    models = models or (env_m.split(",") if env_m else MODEL_ORDER)
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    df, features = load_inputs(input_dir, cfg)
    print(f"rows {len(df)}, features {len(features)}, models {models}, seeds {seeds}")

    # ---- main forward-chaining run
    preds = []
    for m in models:
        for s in seeds:
            preds.append(run_model(df, features, m, s, cfg, out_dir, tag="main"))
    pred = pd.concat(preds, ignore_index=True)
    pc = common_rows(pred)
    print(f"evaluation rows (common to all models): {pc.groupby(['model', 'seed']).size().iloc[0]}")

    comp = comparison_table(pc)
    comp.to_csv(tables / "model_comparison.csv", index=False)
    md = fmt_pm(comp, ["rmse", "mae", "mape", "r2", "rmse_pct", "mae_pct", "total_seconds"],
                {"mape": 2, "r2": 3, "rmse_pct": 2, "mae_pct": 2, "total_seconds": 0})
    md.columns = ["model", "RMSE (kg/ha)", "MAE (kg/ha)", "MAPE (%)", "R²", "RMSE (% of mean)",
                  "MAE (% of mean)", "train+tune time (s)"]
    (tables / "model_comparison.md").write_text(to_markdown(md) + "\n", encoding="utf-8")
    per_fold_table(pc).to_csv(tables / "per_fold_metrics.csv", index=False)
    wil = wilcoxon_vs_baseline(pc)
    wil.to_csv(tables / "wilcoxon_vs_baseline.csv", index=False)
    (tables / "wilcoxon_vs_baseline.md").write_text(to_markdown(wil.round(4), ".4g") + "\n",
                                                    encoding="utf-8")
    diag = diagnostics(pc)
    diag.to_csv(tables / "diagnostics_mahaweli_season_sanity.csv", index=False)
    cols = ["district", "season", "season_year", "y_true", "y_pred", "model", "fold", "seed"]
    out_pred = pred[cols].rename(columns={"season_year": "year"})
    out_pred["lo"], out_pred["hi"] = float("nan"), float("nan")   # filled by conformal (02)
    out_pred.to_csv(tables / "predictions_models.csv", index=False)
    print(comp[["model", "rmse_mean", "rmse_std", "mae_mean", "r2_mean"]].round(3).to_string())
    print(wil[["model", "n_pairs", "median_improvement_kg_ha", "p_holm", "rank_biserial_r"]]
          .round(4).to_string())
    if skip_extra:
        return

    # ---- sensitivity: train without the sanity-flagged rows (LightGBM)
    sens = [run_model(df, features, "lightgbm", s, cfg, out_dir, exclude_flagged_train=True,
                      tag="no_flagged_train") for s in seeds]
    sp = pd.concat(sens + [pred[pred.model == "lightgbm"]], ignore_index=True)
    sp = sp[(sp.sanity_flag == 0) & sp.y_pred.notna()]
    sp["variant"] = sp.tag.map({"main": "trained with flagged rows",
                                "no_flagged_train": "trained without flagged rows"})
    from src.models.metrics import metrics_table
    st = metrics_table(sp, ["variant", "seed"]).groupby("variant")[["n", "rmse", "mae", "r2"]] \
        .agg(["mean", "std"])
    st.columns = [f"{a}_{b}" for a, b in st.columns]
    st.reset_index().to_csv(tables / "sensitivity_sanity_flags.csv", index=False)

    # ---- leave-district-out (fixed parameters, first seed; spatial generalisation)
    ldo = [run_model(df, features, m, seeds[0], cfg, out_dir, splitter="ldo", tag="ldo",
                     fixed_params=p) for m, p in LDO_FIXED.items()]
    lp = common_rows(pd.concat(ldo, ignore_index=True))
    comparison_table(lp).to_csv(tables / "leave_district_out.csv", index=False)

    # ---- random k-fold demonstration
    kd_path = tables / "kfold_vs_forward_demo.csv"
    if not kd_path.exists():
        kfold_demo(df, features, cfg, seeds).to_csv(kd_path, index=False)
    print(pd.read_csv(kd_path).round(3).to_string())
    print("experiment 01 complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed")
    ap.add_argument("--out", default="results/runs/01_models")
    ap.add_argument("--seeds", help="comma list")
    ap.add_argument("--models", help="comma list")
    ap.add_argument("--skip-extra", action="store_true")
    a = ap.parse_args()
    main(Path(a.input), Path(a.out),
         [int(x) for x in a.seeds.split(",")] if a.seeds else None,
         a.models.split(",") if a.models else None, a.skip_extra)
