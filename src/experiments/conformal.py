"""Experiment group 2: split-conformal prediction intervals (forward-chaining, lead 0).

    python -m src.experiments.conformal --input data/processed --out results/runs/02_conformal

Per fold (one test season):
  1. split the training window: proper = older years, calibration = last 3 harvest years
     (src.validation.splitters.calibration_split; nothing from the test season is used);
  2. tune (inner time splits on the PROPER set only) and fit on the proper set;
  3. absolute residuals on the calibration set -> conformal quantile
     q = the ceil((n+1)(1-alpha))-th smallest residual (q = inf if that exceeds n);
  4. interval for each test row: prediction ± q.
Two variants: 'standard' (one q per fold) and 'mondrian_season' (q from calibration rows of
the same season only: Maha and Yala errors differ).
Outputs tables/conformal_*.csv|md, figures/conformal_*.png and preds/conformal__*.parquet.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.plotstyle import INK, MODEL_LABEL, MUTED, SURFACE, line_kw, plt, style
from src.analysis.stats import to_markdown
from src.config import load_config, set_seed
from src.data.features import Calendar
from src.experiments.models import load_inputs
from src.models.registry import BASELINE_COL, MODELS
from src.models.train import ID_COLS, _fit_predict, design_matrix, folds_for, tune
from src.validation.splitters import calibration_split


def conformal_q(resid: np.ndarray, alpha: float) -> float:
    r = np.sort(resid[~np.isnan(resid)])
    n = len(r)
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if n == 0 or k > n else float(r[k - 1])


def run_conformal(df, features, model, seed, cfg, out_dir: Path, log=print) -> pd.DataFrame:
    out = out_dir / "preds" / f"conformal__{model}__seed{seed}.parquet"
    if out.exists():
        log(f"  cached: {out.name}")
        return pd.read_parquet(out)
    cal = Calendar.from_cfg(cfg)
    df = df.reset_index(drop=True)
    X, y = design_matrix(df, features), df["yield_kg_ha"].astype(float)
    b = df[BASELINE_COL].astype(float)
    alphas = sorted(set(cfg["conformal"]["reliability_alphas"]) | {cfg["conformal"]["alpha"]})
    recs, t0 = [], time.time()
    for f in folds_for(df, cal, cfg, "forward", 0):
        proper, calib = calibration_split(df, f.train_idx, cfg["validation"]["calibration_years"])
        te = f.test_idx
        if model == "baseline_last5":
            p_cal, p_te, params = b.loc[calib].to_numpy(), b.loc[te].to_numpy(), {}
        else:
            spec = MODELS[model]
            params = tune(spec, df, X, y, b, proper, seed)
            idx = np.concatenate([calib, te])
            p = _fit_predict(spec, params, seed, X.loc[proper], y.loc[proper].to_numpy(),
                             b.loc[proper].to_numpy(), X.loc[idx], b.loc[idx].to_numpy())
            p_cal, p_te = p[: len(calib)], p[len(calib):]
        res = np.abs(y.loc[calib].to_numpy() - p_cal)
        cal_season = df.loc[calib, "season"].to_numpy()
        part = df.loc[te, ID_COLS].copy()
        part["y_true"], part["y_pred"] = y.loc[te].to_numpy(), p_te
        part["fold"], part["model"], part["seed"] = f.fold_id, model, seed
        part["n_proper"], part["n_calib"] = len(proper), int((~np.isnan(res)).sum())
        part["params"] = json.dumps(params, default=float)
        for a in alphas:
            q_all = conformal_q(res, a)
            part[f"q_standard_{a}"] = q_all
            qs = {s: conformal_q(res[cal_season == s], a) for s in ("Maha", "Yala")}
            part[f"q_mondrian_season_{a}"] = part.season.map(qs).astype(float)
        recs.append(part)
    pred = pd.concat(recs, ignore_index=True)
    pred["total_seconds"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(out, index=False)
    log(f"  conformal {model} seed {seed}: {time.time() - t0:.0f}s")
    return pred


def coverage_long(pred: pd.DataFrame, alphas: list[float]) -> pd.DataFrame:
    rows = []
    for v in ("standard", "mondrian_season"):
        for a in alphas:
            q = pred[f"q_{v}_{a}"]
            rows.append(pred.assign(variant=v, alpha=a, lo=pred.y_pred - q, hi=pred.y_pred + q))
    d = pd.concat(rows, ignore_index=True)
    d["covered"] = (d.y_true >= d.lo) & (d.y_true <= d.hi)
    d["width"] = d.hi - d.lo
    return d


def summarise(d: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = d.groupby(by)
    t = g.agg(n=("covered", "size"), coverage=("covered", "mean"), mean_width=("width", "mean"),
              median_width=("width", "median"), mean_y=("y_true", "mean")).reset_index()
    t["width_pct_of_mean"] = 100 * t.mean_width / t.mean_y
    t["nominal"] = 1 - t.alpha if "alpha" in t else np.nan
    return t


def reliability_figure(t: pd.DataFrame, out: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), facecolor=SURFACE, sharey=True)
    for ax, v in zip(axes, ("standard", "mondrian_season"), strict=True):
        style(ax)
        ax.grid(axis="x", color="#e4e3de", linewidth=0.6)
        ax.plot([0.45, 1], [0.45, 1], color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
        for m in t.model.unique():
            g = t[(t.variant == v) & (t.model == m)].sort_values("nominal")
            kw = line_kw(m)
            ax.plot(g.nominal, g.coverage, marker="o", markersize=4, label=MODEL_LABEL.get(m, m),
                    **kw)
        ax.set_title({"standard": "Standard split conformal",
                      "mondrian_season": "Per-season (Mondrian) conformal"}[v],
                     loc="left", fontsize=10, color=INK)
        ax.set_xlabel("nominal coverage (1 − α)", color=MUTED, fontsize=9)
    axes[0].set_ylabel("observed coverage on test seasons", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def coverage_by_year_figure(t: pd.DataFrame, out: Path, alpha: float, dpi: int) -> None:
    g = t[(t.variant == "standard") & (t.alpha == alpha)].copy()
    g["label"] = g.season.str[0] + " " + g.season_year.astype(str)
    order = g.drop_duplicates("label").sort_values(["season_order"]).label.tolist()
    fig, ax = plt.subplots(figsize=(12, 4), facecolor=SURFACE)
    style(ax)
    ax.axhline(1 - alpha, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
    for m in g.model.unique():
        s = g[g.model == m].set_index("label").reindex(order)
        ax.plot(range(len(order)), s.coverage, marker="o", markersize=3.5,
                label=MODEL_LABEL.get(m, m), **line_kw(m))
    ax.set_xticks(range(len(order)), order, rotation=90)
    ax.set_ylabel(f"coverage of {int(100 * (1 - alpha))}% interval", color=MUTED, fontsize=9)
    ax.set_title("Interval coverage per test season (dashed = target)", loc="left", fontsize=10,
                 color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=3)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def main(input_dir: Path, out_dir: Path) -> None:
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    seed = cfg["followup_seed"]
    models = (os.environ.get("PADDY_MODELS") or ",".join(cfg["conformal"]["models"])).split(",")
    tables, figs = out_dir / "tables", out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    df, features = load_inputs(input_dir, cfg)
    alpha = cfg["conformal"]["alpha"]
    alphas = sorted(set(cfg["conformal"]["reliability_alphas"]) | {alpha})
    pred = pd.concat([run_conformal(df, features, m, seed, cfg, out_dir) for m in models],
                     ignore_index=True)
    # same rows for every model: where all models and the conformal quantiles exist
    ok = pred.y_pred.notna()
    keys = ["district", "season", "season_year"]
    cnt = pred[ok].groupby(keys).model.nunique()
    keep = cnt[cnt == len(models)].index
    pred = pred[ok].set_index(keys).loc[keep].reset_index()
    d = coverage_long(pred, alphas)
    overall = summarise(d, ["model", "variant", "alpha"])
    overall.to_csv(tables / "conformal_coverage.csv", index=False)
    main_t = overall[overall.alpha == alpha][["model", "variant", "n", "coverage", "mean_width",
                                              "median_width", "width_pct_of_mean"]]
    (tables / "conformal_coverage.md").write_text(to_markdown(main_t, ".3f") + "\n",
                                                  encoding="utf-8")
    summarise(d[d.alpha == alpha], ["model", "variant", "alpha", "season"]).to_csv(
        tables / "conformal_coverage_by_season.csv", index=False)
    by_year = summarise(d[d.alpha == alpha], ["model", "variant", "alpha", "season",
                                              "season_year", "season_order"])
    by_year.to_csv(tables / "conformal_coverage_by_test_season.csv", index=False)
    # point accuracy of the proper-set models (they see 3 fewer years than kernel 01's)
    from src.models.metrics import metrics_table
    metrics_table(pred, ["model"]).to_csv(tables / "conformal_point_metrics.csv", index=False)
    a_cols = [f"q_standard_{alpha}", f"q_mondrian_season_{alpha}"]
    p_out = pred[keys + ["y_true", "y_pred", "model", "fold", "seed"] + a_cols].copy()
    p_out["lo"], p_out["hi"] = p_out.y_pred - p_out[a_cols[0]], p_out.y_pred + p_out[a_cols[0]]
    p_out.rename(columns={"season_year": "year"}).to_csv(tables / "predictions_conformal.csv",
                                                         index=False)
    dpi = cfg["figures"]["dpi"]
    reliability_figure(overall, figs / "conformal_reliability.png", dpi)
    coverage_by_year_figure(by_year, figs / "conformal_coverage_by_season.png", alpha, dpi)
    print(main_t.round(3).to_string())
    print("experiment 02 complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed")
    ap.add_argument("--out", default="results/runs/02_conformal")
    a = ap.parse_args()
    main(Path(a.input), Path(a.out))
