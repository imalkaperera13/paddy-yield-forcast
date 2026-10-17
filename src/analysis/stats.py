"""Report tables from predictions: model comparison, Wilcoxon vs baseline, diagnostics,
and the random-k-fold demonstration. Every number here is computed from saved predictions.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.data.features import Calendar
from src.models.metrics import metrics, metrics_table
from src.models.registry import BASELINE_COL
from src.models.train import design_matrix
from src.validation.splitters import forward_chaining

KEY = ["district", "season", "season_year"]


def common_rows(pred: pd.DataFrame) -> pd.DataFrame:
    """Rows where every model AND the baseline have a prediction (same rows for all)."""
    ok = pred.y_pred.notna() & pred.baseline_pred.notna()
    p = pred[ok]
    n_models = p.groupby(["model", "seed"]).ngroups
    counts = p.groupby(KEY).size()
    keep = counts[counts == n_models].index
    return p.set_index(KEY).loc[keep].reset_index()


def comparison_table(pred: pd.DataFrame) -> pd.DataFrame:
    per_seed = metrics_table(pred, ["model", "seed"])
    t = pred.groupby(["model", "seed"]).total_seconds.first().reset_index()
    per_seed = per_seed.merge(t, on=["model", "seed"])
    cols = ["rmse", "mae", "mape", "r2", "rmse_pct", "mae_pct", "bias", "total_seconds"]
    agg = per_seed.groupby("model")[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_seeds"] = per_seed.groupby("model").seed.nunique()
    agg["n_rows"] = per_seed.groupby("model").n.first()
    return agg.reset_index().sort_values("rmse_mean")


def per_fold_table(pred: pd.DataFrame) -> pd.DataFrame:
    t = metrics_table(pred, ["model", "seed", "fold", "season", "season_year"])
    return (t.groupby(["model", "fold", "season", "season_year"])
            [["rmse", "mae", "mape", "r2"]].mean().reset_index())


def wilcoxon_vs_baseline(pred: pd.DataFrame) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank on per district-season absolute errors (model averaged
    over seeds) vs the baseline. Effect size: matched-pairs rank-biserial correlation
    (positive = model has smaller errors). Holm correction across models."""
    base = (pred[pred.model == "baseline_last5"].groupby(KEY)
            .apply(lambda g: np.abs(g.y_true - g.y_pred).mean(), include_groups=False))
    rows = []
    for m in sorted(set(pred.model) - {"baseline_last5"}):
        e = (pred[pred.model == m].groupby(KEY)
             .apply(lambda g: np.abs(g.y_true - g.y_pred).mean(), include_groups=False))
        d = (base - e).dropna()                      # >0 means the model is better
        d = d[d != 0]
        res = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        ranks = pd.Series(np.abs(d)).rank()
        w_pos, w_neg = ranks[d.to_numpy() > 0].sum(), ranks[d.to_numpy() < 0].sum()
        rows.append({"model": m, "n_pairs": len(d), "median_abs_err_model": e.median(),
                     "median_abs_err_baseline": base.loc[e.index].median(),
                     "median_improvement_kg_ha": d.median(), "share_rows_better": (d > 0).mean(),
                     "W": res.statistic, "p_value": res.pvalue,
                     "rank_biserial_r": (w_pos - w_neg) / (w_pos + w_neg)})
    out = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    k = len(out)
    adj = np.minimum(1, out.p_value.to_numpy() * (k - np.arange(k)))   # Holm step-down
    out["p_holm"] = np.maximum.accumulate(adj)
    return out


def diagnostics(pred: pd.DataFrame) -> pd.DataFrame:
    """Mahaweli diagnostics (a)/(b)/(c) and per-season metrics, mean over seeds."""
    subsets = {
        "(a) all rows": pred,
        "(b) excl. post-merge rows of affected districts": pred[pred.diag_post_merge_affected == 0],
        "(c) affected districts, before merge": pred[(pred.mahaweli_affected == 1)
                                                     & (pred.mahaweli_merged == 0)],
        "(c) affected districts, after merge": pred[(pred.mahaweli_affected == 1)
                                                    & (pred.mahaweli_merged == 1)],
        "(c) other districts, before merge": pred[(pred.mahaweli_affected == 0)
                                                  & (pred.mahaweli_merged == 0)],
        "(c) other districts, after merge": pred[(pred.mahaweli_affected == 0)
                                                 & (pred.mahaweli_merged == 1)],
        "season = Maha": pred[pred.season == "Maha"],
        "season = Yala": pred[pred.season == "Yala"],
        "sanity_flag = 0 rows": pred[pred.sanity_flag == 0],
    }
    rows = []
    for name, p in subsets.items():
        if len(p) == 0:
            continue
        t = metrics_table(p, ["model", "seed"]).groupby("model")[["n", "rmse", "mae", "r2"]].mean()
        t["subset"] = name
        rows.append(t.reset_index())
    return pd.concat(rows, ignore_index=True)


def kfold_demo(df: pd.DataFrame, features: list[str], cfg: dict, seeds: list[int],
               log=print) -> pd.DataFrame:
    """Same LightGBM (fixed parameters, no tuning) scored two ways on the SAME rows:
    random 5-fold (shuffled; training contains future seasons) vs forward-chaining.
    For the documented demonstration only - never used for model selection."""
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import KFold

    cal = Calendar.from_cfg(cfg)
    df = df.reset_index(drop=True)
    X, y = design_matrix(df, features), df.yield_kg_ha.to_numpy(float)
    folds = list(forward_chaining(df, cal, cfg["validation"]["min_train_years"]))
    eval_idx = np.concatenate([f.test_idx for f in folds])
    params = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=-1)
    rows = []
    for seed in seeds:
        t0 = time.time()
        p_fc = pd.Series(np.nan, index=df.index)
        for f in folds:
            m = LGBMRegressor(random_state=seed, **params).fit(X.loc[f.train_idx], y[f.train_idx])
            p_fc.loc[f.test_idx] = m.predict(X.loc[f.test_idx])
        p_rk = pd.Series(np.nan, index=df.index)
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
            m = LGBMRegressor(random_state=seed, **params).fit(X.iloc[tr], y[tr])
            p_rk.iloc[te] = m.predict(X.iloc[te])
        for scheme, p in (("forward-chaining (honest)", p_fc), ("random 5-fold (leaky)", p_rk)):
            rows.append({"scheme": scheme, "seed": seed,
                         **metrics(y[eval_idx], p.loc[eval_idx].to_numpy())})
        log(f"  k-fold demo seed {seed}: {time.time() - t0:.0f}s")
    t = pd.DataFrame(rows)
    out = t.groupby("scheme")[["n", "rmse", "mae", "r2", "rmse_pct"]].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def baseline_check(df: pd.DataFrame) -> None:
    assert BASELINE_COL in df, "baseline column missing"


def to_markdown(df: pd.DataFrame, floatfmt: str = ".1f") -> str:
    def fmt(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else format(v, floatfmt)
        return str(v)
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])
