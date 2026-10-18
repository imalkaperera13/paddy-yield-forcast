"""Experiment group 4: feature-group ablation and SHAP (random forest, forward-chaining, lead 0).

    python -m src.experiments.ablation --input data/processed --out results/runs/04_ablation_shap

Ablation variants (groups from feature_list.json):
  all; lagged_yield only; lagged_yield + one other group; all minus one group;
  no lagged_yield (satellite + climate only).
SHAP: for the 'all' variant, each fold's model is refitted with the parameters that fold chose
(stored in its predictions) and explained on that fold's test rows only (TreeExplainer).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.analysis.plotstyle import INK, MUTED, SLOTS, SURFACE, plt, style
from src.analysis.stats import to_markdown
from src.config import load_config, set_seed
from src.data.features import Calendar
from src.experiments.models import load_inputs
from src.models.metrics import metrics_table
from src.models.registry import MODELS
from src.models.train import design_matrix, folds_for, run_model

KEYS = ["district", "season", "season_year"]


def variants(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    allf = [c for g in groups.values() for c in g]
    lag = groups["lagged_yield"]
    others = [g for g in groups if g != "lagged_yield"]
    v = {"all": allf, "lagged_yield_only": lag}
    for g in others:
        v[f"lagged_yield+{g}"] = lag + groups[g]
    for g in others:
        v[f"all_minus_{g}"] = [c for c in allf if c not in groups[g]]
    v["no_lagged_yield"] = [c for g in others for c in groups[g]]
    return v


def shap_values(df, features, pred_all: pd.DataFrame, cfg, seed: int, log=print):
    import shap
    cal = Calendar.from_cfg(cfg)
    df = df.reset_index(drop=True)
    X, y = design_matrix(df, features), df["yield_kg_ha"].astype(float)
    spec = MODELS["random_forest"]
    params_by_fold = pred_all.groupby("fold").params.first().map(json.loads)
    out = []
    for f in folds_for(df, cal, cfg, "forward", 0):
        pipe = spec.build(params_by_fold[f.fold_id], seed)
        pipe.fit(X.loc[f.train_idx], y.loc[f.train_idx].to_numpy())
        imp, rf = pipe[0], pipe[-1]
        Xt = pd.DataFrame(imp.transform(X.loc[f.test_idx]), columns=X.columns)
        sv = shap.TreeExplainer(rf).shap_values(Xt, check_additivity=False)
        s = pd.DataFrame(sv, columns=X.columns)
        s[KEYS] = df.loc[f.test_idx, KEYS].to_numpy()
        s["fold"] = f.fold_id
        out.append(s)
    log(f"  SHAP done for {len(out)} folds")
    return pd.concat(out, ignore_index=True)


def shap_figure(imp: pd.DataFrame, group_of: dict, out: Path, dpi: int, top: int = 20) -> None:
    gcol = {g: SLOTS[i] for i, g in enumerate(sorted(set(group_of.values())))}
    t = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e4e3de", linewidth=0.6)
    ax.barh(t.feature, t.mean_abs_shap, height=0.6,
            color=[gcol.get(group_of.get(f, ""), MUTED) for f in t.feature])
    from matplotlib.patches import Patch
    shown = [g for g in gcol if g in {group_of.get(f) for f in t.feature}]
    ax.legend(handles=[Patch(color=gcol[g], label=g) for g in shown], frameon=False, fontsize=8,
              loc="lower right")
    ax.set_xlabel("mean |SHAP| (kg/ha)", color=MUTED, fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(f"Random forest: top {top} features by mean |SHAP| over all test rows",
                 loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def redraw_shap_figure(imp_csv: Path, out: Path, dpi: int = 300) -> None:
    """Re-plot the SHAP bar chart from a saved shap_feature_importance.csv (no recomputation)."""
    imp = pd.read_csv(imp_csv)
    shap_figure(imp, dict(zip(imp.feature, imp.group, strict=True)), out, dpi)


def main(input_dir: Path, out_dir: Path) -> None:
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    seed = cfg["followup_seed"]
    tables, figs = out_dir / "tables", out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    df, _ = load_inputs(input_dir, cfg)
    groups = json.loads((input_dir / "feature_list.json").read_text(encoding="utf-8"))["groups"]
    vs = variants(groups)
    only = os.environ.get("PADDY_VARIANTS")
    if only:
        vs = {k: v for k, v in vs.items() if k in only.split(",")}
    preds = [run_model(df, vs["all"][:1], "baseline_last5", seed, cfg, out_dir, tag="abl_baseline")
             .assign(variant="baseline (last-5 mean)", n_features=0)]
    for name, feats in vs.items():
        p = run_model(df, feats, "random_forest", seed, cfg, out_dir, tag=f"abl_{name}")
        preds.append(p.assign(variant=name, n_features=len(feats)))
    pred = pd.concat(preds, ignore_index=True)
    ok = pred.y_pred.notna() & pred.baseline_pred.notna()
    n = pred.variant.nunique()
    cnt = pred[ok].groupby(KEYS).variant.nunique()
    pc = pred[ok].set_index(KEYS).loc[cnt[cnt == n].index].reset_index()
    t = metrics_table(pc, ["variant"]).merge(pc.groupby("variant").n_features.first().reset_index())
    t = t.sort_values("rmse")
    t.to_csv(tables / "ablation.csv", index=False)
    (tables / "ablation.md").write_text(
        to_markdown(t[["variant", "n_features", "n", "rmse", "mae", "r2", "bias"]]) + "\n",
        encoding="utf-8")
    rows = []
    base = pc[pc.variant.str.startswith("baseline")].set_index(KEYS)
    full = pc[pc.variant == "all"].set_index(KEYS)
    for v, g in pc[~pc.variant.str.startswith("baseline")].groupby("variant"):
        g = g.set_index(KEYS)
        for ref_name, ref in (("baseline", base), ("all", full)):
            if v == "all" and ref_name == "all":
                continue
            r = ref.loc[g.index]
            d = ((r.y_pred - r.y_true).abs() - (g.y_pred - g.y_true).abs()).to_numpy()
            d = d[d != 0]
            rk = pd.Series(abs(d)).rank()
            rows.append({"variant": v, "vs": ref_name, "n_pairs": len(d),
                         "p_value": float(wilcoxon(d).pvalue),
                         "rank_biserial_r": float((rk[d > 0].sum() - rk[d < 0].sum()) / rk.sum()),
                         "median_abs_err_gain_kg_ha": float(np.median(d))})
    wt = pd.DataFrame(rows)
    order = wt.p_value.sort_values().index
    k, run = len(wt), 0.0
    wt["p_holm"] = np.nan
    for i, ix in enumerate(order):
        run = max(run, min(1.0, (k - i) * wt.p_value[ix]))
        wt.loc[ix, "p_holm"] = run
    wt.to_csv(tables / "ablation_wilcoxon.csv", index=False)
    (tables / "ablation_wilcoxon.md").write_text(to_markdown(wt, ".4g") + "\n", encoding="utf-8")
    print(t[["variant", "n_features", "n", "rmse", "r2"]].round(3).to_string())

    # ---- SHAP on the 'all' random forest
    if "all" in vs and not os.environ.get("PADDY_SKIP_SHAP"):
        sp = out_dir / "shap_values.parquet"
        if sp.exists():
            sv = pd.read_parquet(sp)
        else:
            all_pred = pred[pred.variant == "all"]
            sv = shap_values(df, vs["all"], all_pred, cfg, seed)
            sv.to_parquet(sp, index=False)
        feat_cols = [c for c in sv.columns if c not in KEYS + ["fold"]]
        imp = sv[feat_cols].abs().mean().sort_values(ascending=False).rename("mean_abs_shap")
        group_of = {c: g for g, cs in groups.items() for c in cs}
        group_of["is_maha"] = "season flag"
        imp = imp.reset_index().rename(columns={"index": "feature"})
        imp["group"] = imp.feature.map(group_of)
        imp.to_csv(tables / "shap_feature_importance.csv", index=False)
        gi = imp.groupby("group").mean_abs_shap.agg(["sum", "count"]).reset_index() \
            .rename(columns={"sum": "sum_mean_abs_shap", "count": "n_features"}) \
            .sort_values("sum_mean_abs_shap", ascending=False)
        gi["share"] = gi.sum_mean_abs_shap / gi.sum_mean_abs_shap.sum()
        gi.to_csv(tables / "shap_group_importance.csv", index=False)
        (tables / "shap_group_importance.md").write_text(to_markdown(gi, ".3f") + "\n",
                                                         encoding="utf-8")
        shap_figure(imp, group_of, figs / "shap_top_features.png", cfg["figures"]["dpi"])
        print(gi.round(3).to_string())
    print("experiment 04 complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed")
    ap.add_argument("--out", default="results/runs/04_ablation_shap")
    a = ap.parse_args()
    main(Path(a.input), Path(a.out))
