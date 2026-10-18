"""Experiment group 5: fit and save the final models on all published labels.

    python -m src.experiments.final_model --input data/processed --out results/runs/05_final_model

Saved (out/models/):
  rf_all.joblib            random forest, all 120 features + season flag (main model, kernel 01)
  rf_lagged_modis.joblib   random forest, lagged yield + MODIS (best ablation variant, kernel 04)
  final_models.json        features, tuned parameters, training rows, split-conformal q (α from
                           config, calibration = last 3 harvest years, model fitted on older
                           years only), data SHA-256 and code commit.
The baseline needs no model: it is the column y_last5_same.
These models are for forecasting NEW seasons; every reported metric comes from the
forward-chaining experiments, never from these fits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, set_seed
from src.experiments.ablation import variants
from src.experiments.conformal import conformal_q
from src.experiments.models import load_inputs
from src.models.registry import BASELINE_COL, MODELS
from src.models.train import _fit_predict, design_matrix, tune
from src.validation.splitters import calibration_split


def fit_one(df, features, seed, cfg) -> tuple[object, dict]:
    spec = MODELS["random_forest"]
    X, y = design_matrix(df, features), df["yield_kg_ha"].astype(float)
    b = df[BASELINE_COL].astype(float)
    idx = np.arange(len(df))
    params = tune(spec, df, X, y, b, idx, seed)
    proper, calib = calibration_split(df, idx, cfg["validation"]["calibration_years"])
    p_cal = _fit_predict(spec, params, seed, X.loc[proper], y.loc[proper].to_numpy(),
                         b.loc[proper].to_numpy(), X.loc[calib], b.loc[calib].to_numpy())
    alpha = cfg["conformal"]["alpha"]
    q = conformal_q(np.abs(y.loc[calib].to_numpy() - p_cal), alpha)
    model = spec.build(params, seed)
    model.fit(X, y.to_numpy())
    hy = df["harvest_year"]
    meta = {"features": list(X.columns), "params": params, "seed": seed,
            "n_train": int(len(df)), "train_harvest_years": [int(hy.min()), int(hy.max())],
            "conformal": {"alpha": alpha, "q_kg_ha": q, "n_calibration": int(len(calib)),
                          "calibration_harvest_years": sorted(int(v) for v in
                                                              hy.iloc[calib].unique())}}
    return model, meta


def main(input_dir: Path, out_dir: Path, commit: str | None = None) -> None:
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    seed = cfg["followup_seed"]
    df, _ = load_inputs(input_dir, cfg)
    df = df.reset_index(drop=True)
    groups = json.loads((input_dir / "feature_list.json").read_text(encoding="utf-8"))["groups"]
    vs = variants(groups)
    mdir = out_dir / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256((input_dir / "master_table.parquet").read_bytes()).hexdigest()
    import sklearn
    out = {"data_sha256": sha, "code_commit": commit or os.environ.get("PADDY_COMMIT", "unknown"),
           "sklearn": sklearn.__version__, "baseline": f"column {BASELINE_COL}", "models": {}}
    for name, key in (("rf_all", "all"), ("rf_lagged_modis", "lagged_yield+modis")):
        model, meta = fit_one(df, vs[key], seed, cfg)
        joblib.dump(model, mdir / f"{name}.joblib")
        out["models"][name] = {"file": f"{name}.joblib", "feature_set": key, **meta}
        print(f"{name}: params {meta['params']}, q = {meta['conformal']['q_kg_ha']:.1f} kg/ha, "
              f"n_train {meta['n_train']}")
    (mdir / "final_models.json").write_text(json.dumps(out, indent=2, default=float),
                                            encoding="utf-8")
    print("experiment 05 complete")


def predict(model_path: Path, meta_path: Path, rows: pd.DataFrame) -> pd.DataFrame:
    """Forecast new district-seasons with a saved model: point, lo, hi."""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    m = meta["models"][model_path.stem]
    X = rows.assign(is_maha=(rows["season"] == "Maha").astype(float))[m["features"]].astype(float)
    pred = joblib.load(model_path).predict(X)
    q = m["conformal"]["q_kg_ha"]
    return rows[["district", "season", "season_year"]].assign(y_pred=pred, lo=pred - q,
                                                             hi=pred + q)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed")
    ap.add_argument("--out", default="results/runs/05_final_model")
    a = ap.parse_args()
    main(Path(a.input), Path(a.out))
