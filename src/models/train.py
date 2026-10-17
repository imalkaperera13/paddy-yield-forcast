"""Train/evaluate every model through the SAME leakage-safe splits.

For each fold (one test season):
  1. tune on the training window only (inner expanding-window years, src.validation),
  2. refit on the whole training window with the chosen parameters,
  3. predict the test season.
Results are saved per (model, seed, tag) so a stopped Kaggle session resumes where it left off.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.features import Calendar
from src.models.metrics import metrics
from src.models.registry import BASELINE_COL, MODELS, ModelSpec
from src.validation.splitters import forward_chaining, inner_time_splits, leave_district_out

ID_COLS = ["district", "season", "season_label", "season_year", "season_order",
           "mahaweli_merged", "mahaweli_affected", "diag_post_merge_affected", "sanity_flag"]


def design_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df[features].astype(float).copy()
    X["is_maha"] = (df["season"] == "Maha").astype(float)
    return X


def _fit_predict(spec: ModelSpec, params: dict, seed: int, X_tr, y_tr, b_tr, X_te, b_te):
    model = spec.build(params, seed)
    if spec.residual:
        ok = ~np.isnan(b_tr)
        model.fit(X_tr[ok], y_tr[ok] - b_tr[ok])
        return b_te + model.predict(X_te)
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def tune(spec: ModelSpec, df, X, y, b, train_idx, seed, n_inner: int = 3) -> dict:
    if len(spec.grid) == 1:
        return spec.grid[0]
    best, best_mae = None, np.inf
    splits = inner_time_splits(df, train_idx, n_inner)
    for params in spec.grid:
        errs = []
        for tr, va in splits:
            p = _fit_predict(spec, params, seed, X.loc[tr], y.loc[tr].to_numpy(),
                             b.loc[tr].to_numpy(), X.loc[va], b.loc[va].to_numpy())
            ok = ~np.isnan(p)
            errs.append(np.mean(np.abs(y.loc[va].to_numpy()[ok] - p[ok])))
        mae = float(np.mean(errs))
        if mae < best_mae:
            best, best_mae = params, mae
    return best


def folds_for(df, cal, cfg, splitter: str, lead: int):
    """PADDY_SMOKE=1 keeps only the last 3 test seasons: a fast code-path check, never results."""
    k = cfg["validation"]["min_train_years"]
    if splitter == "forward":
        folds = list(forward_chaining(df, cal, k, lead))
    elif splitter == "ldo":
        folds = list(leave_district_out(df, cal, k, lead))
    else:
        raise ValueError(splitter)
    if os.environ.get("PADDY_SMOKE"):
        last = sorted({f.test_order for f in folds})[-3:]
        folds = [f for f in folds if f.test_order in last]
    return folds


def run_model(df: pd.DataFrame, features: list[str], model: str, seed: int, cfg: dict,
              out_dir: Path, splitter: str = "forward", lead: int = 0,
              exclude_flagged_train: bool = False, tag: str = "main",
              fixed_params: dict | None = None, log=print) -> pd.DataFrame:
    """Predictions for every test row of every fold. Cached in out_dir/preds/."""
    out = out_dir / "preds" / f"{tag}__{model}__seed{seed}.parquet"
    if out.exists():
        log(f"  cached: {out.name}")
        return pd.read_parquet(out)
    cal = Calendar.from_cfg(cfg)
    df = df.reset_index(drop=True)
    X = design_matrix(df, features)
    y = df["yield_kg_ha"].astype(float)
    b = df[BASELINE_COL].astype(float)
    recs, t0 = [], time.time()
    for f in folds_for(df, cal, cfg, splitter, lead):
        tr = f.train_idx
        if exclude_flagged_train:
            tr = tr[df.loc[tr, "sanity_flag"].to_numpy() == 0]
        te = f.test_idx
        t_fold = time.time()
        if model == "baseline_last5":
            pred, params = b.loc[te].to_numpy(), {}
        else:
            spec = MODELS[model]
            params = fixed_params if fixed_params is not None else tune(spec, df, X, y, b, tr,
                                                                         seed)
            pred = _fit_predict(spec, params, seed, X.loc[tr], y.loc[tr].to_numpy(),
                                b.loc[tr].to_numpy(), X.loc[te], b.loc[te].to_numpy())
        part = df.loc[te, ID_COLS].copy()
        part["y_true"], part["y_pred"] = y.loc[te].to_numpy(), pred
        part["baseline_pred"] = b.loc[te].to_numpy()
        part["fold"], part["model"], part["seed"] = f.fold_id, model, seed
        part["held_out_district"] = f.held_out_district
        part["params"] = json.dumps(params, default=float)
        part["fit_seconds"] = time.time() - t_fold
        part["n_train"] = len(tr)
        recs.append(part)
    pred = pd.concat(recs, ignore_index=True)
    pred["tag"], pred["splitter"], pred["lead_weeks"] = tag, splitter, lead
    pred["total_seconds"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(out, index=False)
    ok = pred.y_pred.notna() & pred.baseline_pred.notna()
    m = metrics(pred.y_true[ok], pred.y_pred[ok])
    log(f"  {tag} {model} seed {seed}: RMSE {m['rmse']:.0f} MAE {m['mae']:.0f} "
        f"R2 {m['r2']:.3f} ({time.time() - t0:.0f}s)")
    _mlflow_log(cfg, out_dir, tag, model, seed, splitter, lead, m, time.time() - t0)
    return pred


def _mlflow_log(cfg, out_dir, tag, model, seed, splitter, lead, m, secs) -> None:
    """Always append to runs_log.jsonl; also log to MLflow (SQLite store) if available.
    Logging problems never stop a training run."""
    rec = {"tag": tag, "model": model, "seed": seed, "splitter": splitter, "lead_weeks": lead,
           "seconds": round(secs, 1), **{k: v for k, v in m.items()}}
    with open(out_dir / "runs_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    try:
        import mlflow
        uri = f"sqlite:///{(out_dir / 'mlflow.db').as_posix()}"
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        mlflow.set_experiment("paddy-yield")
        with mlflow.start_run(run_name=f"{tag}-{model}-s{seed}"):
            mlflow.log_params({"tag": tag, "model": model, "seed": seed,
                               "splitter": splitter, "lead_weeks": lead})
            mlflow.log_metrics({k: float(v) for k, v in m.items()
                                if isinstance(v, (int, float))} | {"seconds": secs})
    except Exception as e:  # noqa: BLE001
        print(f"  (mlflow logging skipped: {type(e).__name__}: {str(e)[:120]})")
