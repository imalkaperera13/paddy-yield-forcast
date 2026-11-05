"""Step 6: static paddy mask vs pilot WorldCover-cropland features, 7 study districts,
forecast leads 12 / 8 / 4 weeks before the reference harvest.

    python -m src.experiments.masked_compare models    --input DIR --out DIR   # Kaggle kernel 06a
    python -m src.experiments.masked_compare followups --input DIR --out DIR   # Kaggle kernel 06b
    python -m src.experiments.masked_compare report    --runs DIR [DIR ...]    # local, no training

Inputs (private Kaggle dataset paddycast-masked): master_masked.parquet,
master_unmasked_7d.parquet (identical rows), feature_list.json.

Stage `models`
  forward chaining, every lead x mask x model x seed: baseline_last5, ridge, random_forest,
  lightgbm, xgboost (tuning grids and inner time splits from src.models.registry/train);
  leave-district-out with fixed, pre-chosen parameters (first seed).
  Masked and unmasked use the SAME folds, seeds and grids: only the feature values differ.
Stage `followups`
  split conformal (standard and Mondrian by season) for cfg.conformal.models;
  random-forest feature-group ablation and SHAP (TreeExplainer on each fold's test rows).
Stage `report`
  tables from the saved predictions only: model comparison, masked vs unmasked and model vs
  baseline Wilcoxon signed-rank tests with Holm correction, earliness, leave-district-out,
  conformal coverage by season, ablation, SHAP group shares.
Every stage caches per unit (out/preds/*.parquet), so a stopped session resumes.
Environment filters: PADDY_LEADS, PADDY_MASKS, PADDY_MODELS, PADDY_SEEDS, PADDY_SMOKE.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from src.config import load_config, set_seed
from src.data.features import Calendar
from src.experiments.ablation import variants
from src.experiments.conformal import conformal_q
from src.models.metrics import metrics
from src.models.registry import BASELINE_COL, MODELS
from src.models.train import (
    ID_COLS,
    _fit_predict,
    design_matrix,
    folds_for,
    run_model,
    tune,
)
from src.validation.splitters import calibration_split

MASKS = {"masked": "master_masked.parquet", "unmasked": "master_unmasked_7d.parquet"}
LEADS = (12, 8, 4)
MODEL_ORDER = ["baseline_last5", "ridge", "random_forest", "lightgbm", "xgboost"]
# leave-district-out: fixed parameters chosen before any run (pilot LDO values for ridge and
# LightGBM; for RF and XGBoost the lowest-capacity point of their tuning grids)
LDO_FIXED = {"baseline_last5": {}, "ridge": {"alpha": 10.0},
             "random_forest": {"min_samples_leaf": 15, "max_features": 0.5},
             "lightgbm": {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 7},
             "xgboost": {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 3}}
KEY = ["district", "season", "season_year"]
TAG_RE = re.compile(r"^(?P<kind>ldo_|conformal_|abl_)?(?P<mask>masked|unmasked)_L(?P<lead>\d+)"
                    r"(?:_(?P<variant>.+))?$")


# ---------------------------------------------------------------- inputs
def load_tables(input_dir: Path, cfg: dict) -> tuple[dict[str, pd.DataFrame], dict]:
    tabs = {m: pd.read_parquet(input_dir / f) for m, f in MASKS.items()}
    k = ["district", "season", "season_year", "lead_weeks"]
    if not (tabs["masked"][k].values == tabs["unmasked"][k].values).all():
        raise RuntimeError("masked and unmasked tables do not have identical rows")
    groups = json.loads((input_dir / "feature_list.json").read_text(encoding="utf-8"))["groups"]
    feats = [c for g in groups.values() for c in g]
    bad = set(feats) & set(cfg["forbidden_features"])
    if bad:
        raise RuntimeError(f"forbidden features in feature list: {bad}")
    for m, t in tabs.items():
        miss = set(feats) - set(t.columns)
        if miss:
            raise RuntimeError(f"{m}: features missing {sorted(miss)}")
    return tabs, groups


def lead_table(tabs: dict, mask: str, lead: int) -> pd.DataFrame:
    t = tabs[mask]
    return t[t.lead_weeks == lead].sort_values(KEY).reset_index(drop=True)


def _env_list(name: str, default: list, cast=str) -> list:
    v = os.environ.get(name)
    return [cast(x) for x in v.split(",")] if v else list(default)


def _all_features(groups: dict) -> list[str]:
    return [c for g in groups.values() for c in g]


# ---------------------------------------------------------------- stage: models
def stage_models(input_dir: Path, out: Path) -> None:
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    tabs, groups = load_tables(input_dir, cfg)
    feats = _all_features(groups)
    leads = _env_list("PADDY_LEADS", LEADS, int)
    masks = _env_list("PADDY_MASKS", MASKS)
    models = _env_list("PADDY_MODELS", MODEL_ORDER)
    seeds = _env_list("PADDY_SEEDS", cfg["project"]["seeds_for_spread"], int)
    print(f"leads {leads} masks {masks} models {models} seeds {seeds} features {len(feats)}")
    t0 = time.time()
    for lead in leads:
        for mask in masks:
            df = lead_table(tabs, mask, lead)
            for model in models:
                for seed in seeds:
                    run_model(df, feats, model, seed, cfg, out, splitter="forward", lead=lead,
                              tag=f"{mask}_L{lead}")
            print(f"== forward {mask} L{lead} done, {time.time() - t0:.0f}s so far", flush=True)
    for lead in leads:
        for mask in masks:
            df = lead_table(tabs, mask, lead)
            for model in models:
                run_model(df, feats, model, seeds[0], cfg, out, splitter="ldo", lead=lead,
                          tag=f"ldo_{mask}_L{lead}", fixed_params=LDO_FIXED[model])
            print(f"== ldo {mask} L{lead} done, {time.time() - t0:.0f}s so far", flush=True)
    print("stage models complete")


# ---------------------------------------------------------------- stage: followups
def run_conformal_lead(df, features, model, seed, cfg, out_dir: Path, lead: int, tag: str,
                       log=print) -> pd.DataFrame:
    """The pilot's split-conformal procedure (src.experiments.conformal) at forecast lead
    `lead`: per fold, proper = training window minus its last calibration_years harvest years,
    calibration = those years; fit on proper only; q from calibration |residuals|."""
    out = out_dir / "preds" / f"conformal_{tag}__{model}__seed{seed}.parquet"
    if out.exists():
        log(f"  cached: {out.name}")
        return pd.read_parquet(out)
    cal = Calendar.from_cfg(cfg)
    df = df.reset_index(drop=True)
    X, y = design_matrix(df, features), df["yield_kg_ha"].astype(float)
    b = df[BASELINE_COL].astype(float)
    alphas = sorted(set(cfg["conformal"]["reliability_alphas"]) | {cfg["conformal"]["alpha"]})
    recs, t0 = [], time.time()
    for f in folds_for(df, cal, cfg, "forward", lead):
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
        part["baseline_pred"] = b.loc[te].to_numpy()
        part["fold"], part["model"], part["seed"] = f.fold_id, model, seed
        part["n_proper"], part["n_calib"] = len(proper), int((~np.isnan(res)).sum())
        for s in ("Maha", "Yala"):
            part[f"n_calib_{s}"] = int((~np.isnan(res[cal_season == s])).sum())
        part["calib_last_harvest"] = int(df.loc[calib, "harvest_year"].max())
        part["params"] = json.dumps(params, default=float)
        for a in alphas:
            part[f"q_standard_{a}"] = conformal_q(res, a)
            qs = {s: conformal_q(res[cal_season == s], a) for s in ("Maha", "Yala")}
            part[f"q_mondrian_season_{a}"] = part.season.map(qs).astype(float)
        recs.append(part)
    pred = pd.concat(recs, ignore_index=True)
    pred["tag"], pred["lead_weeks"] = f"conformal_{tag}", lead
    pred["total_seconds"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(out, index=False)
    log(f"  conformal {tag} {model} seed {seed}: {time.time() - t0:.0f}s")
    return pred


def shap_values_lead(df, features, pred_all: pd.DataFrame, cfg, seed: int, lead: int):
    """Pilot SHAP (src.experiments.ablation.shap_values) at forecast lead `lead`: each fold's
    RF is refitted with that fold's chosen parameters and explained on its test rows only."""
    import shap
    cal = Calendar.from_cfg(cfg)
    df = df.reset_index(drop=True)
    X, y = design_matrix(df, features), df["yield_kg_ha"].astype(float)
    spec = MODELS["random_forest"]
    params_by_fold = pred_all.groupby("fold").params.first().map(json.loads)
    out = []
    for f in folds_for(df, cal, cfg, "forward", lead):
        pipe = spec.build(params_by_fold[f.fold_id], seed)
        pipe.fit(X.loc[f.train_idx], y.loc[f.train_idx].to_numpy())
        imp, rf = pipe[0], pipe[-1]
        # columns with no observed training value (e.g. blocks after the cutoff at lead 12)
        # are dropped by the imputer; their SHAP value is 0 by construction
        kept = list(imp.get_feature_names_out(X.columns))
        Xt = pd.DataFrame(imp.transform(X.loc[f.test_idx]), columns=kept)
        sv = shap.TreeExplainer(rf).shap_values(Xt, check_additivity=False)
        s = pd.DataFrame(0.0, index=range(len(Xt)), columns=X.columns)
        s[kept] = sv
        s[KEY] = df.loc[f.test_idx, KEY].to_numpy()
        s["fold"] = f.fold_id
        out.append(s)
    return pd.concat(out, ignore_index=True)


def stage_followups(input_dir: Path, out: Path) -> None:
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    tabs, groups = load_tables(input_dir, cfg)
    feats = _all_features(groups)
    seed = cfg["followup_seed"]
    leads = _env_list("PADDY_LEADS", LEADS, int)
    masks = _env_list("PADDY_MASKS", MASKS)
    t0 = time.time()
    for lead in leads:
        for mask in masks:
            df = lead_table(tabs, mask, lead)
            for model in _env_list("PADDY_MODELS", cfg["conformal"]["models"]):
                run_conformal_lead(df, feats, model, seed, cfg, out, lead, f"{mask}_L{lead}")
            print(f"== conformal {mask} L{lead} done, {time.time() - t0:.0f}s", flush=True)
    if os.environ.get("PADDY_SKIP_ABLATION"):
        return
    vs = variants(groups)
    for lead in leads:
        for mask in masks:
            df = lead_table(tabs, mask, lead)
            tag = f"abl_{mask}_L{lead}"
            run_model(df, feats, "baseline_last5", seed, cfg, out, lead=lead,
                      tag=f"{tag}_baseline")
            for name, fs in vs.items():
                p = run_model(df, fs, "random_forest", seed, cfg, out, lead=lead,
                              tag=f"{tag}_{name}")
                if name == "all" and not os.environ.get("PADDY_SKIP_SHAP"):
                    sp = out / "shap" / f"shap_{mask}_L{lead}.parquet"
                    if not sp.exists():
                        sp.parent.mkdir(parents=True, exist_ok=True)
                        shap_values_lead(df, fs, p, cfg, seed, lead).to_parquet(sp, index=False)
            print(f"== ablation+SHAP {mask} L{lead} done, {time.time() - t0:.0f}s", flush=True)
    print("stage followups complete")


# ---------------------------------------------------------------- stage: report (no training)
def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    k = len(p)
    adj = np.empty(k)
    run = 0.0
    for i, ix in enumerate(order):
        run = max(run, min(1.0, (k - i) * p[ix]))
        adj[ix] = run
    return adj


def paired_test(d: pd.Series) -> dict:
    """d > 0 means the first-named option has smaller absolute error. Zeros dropped
    (Wilcoxon 'wilcox' convention). Effect: matched-pairs rank-biserial correlation."""
    n_all = len(d)
    d = d[d != 0].to_numpy()
    if len(d) < 2:
        return {"n_pairs": n_all, "n_nonzero": len(d), "median_gain_kg_ha": np.nan,
                "share_better": np.nan, "W": np.nan, "p_value": np.nan, "rank_biserial_r": np.nan}
    res = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    rk = pd.Series(np.abs(d)).rank().to_numpy()
    return {"n_pairs": n_all, "n_nonzero": len(d), "median_gain_kg_ha": float(np.median(d)),
            "share_better": float((d > 0).mean()), "W": float(res.statistic),
            "p_value": float(res.pvalue),
            "rank_biserial_r": float((rk[d > 0].sum() - rk[d < 0].sum()) / rk.sum())}


def _with_holm(rows: list[dict], family: str) -> pd.DataFrame:
    t = pd.DataFrame(rows)
    ok = t.p_value.notna()
    t["p_holm"] = np.nan
    t.loc[ok, "p_holm"] = holm(t.loc[ok, "p_value"].to_numpy())
    t["holm_family"] = family
    t["family_size"] = int(ok.sum())
    return t


def load_preds(run_dirs: list[Path]) -> pd.DataFrame:
    parts = []
    for d in run_dirs:
        for f in sorted((d / "preds").glob("*.parquet")):
            p = pd.read_parquet(f)
            m = TAG_RE.match(str(p.tag.iloc[0]))
            if not m:
                continue
            p["kind"] = (m["kind"] or "forward_").rstrip("_")
            p["mask"], p["lead"] = m["mask"], int(m["lead"])
            p["variant"] = m["variant"]
            parts.append(p)
    if not parts:
        raise SystemExit(f"no prediction files found under {run_dirs}")
    return pd.concat(parts, ignore_index=True)


def common(p: pd.DataFrame, units: list[str]) -> pd.DataFrame:
    """Rows (district x season x year) predicted by every unit, with a baseline value."""
    ok = p.y_pred.notna() & np.isfinite(p.y_pred)
    if "baseline_pred" in p:
        ok &= p.baseline_pred.notna()
    q = p[ok]
    n = q.groupby(units).ngroups
    cnt = q.groupby(KEY)[units].apply(lambda g: len(g.drop_duplicates()))
    keep = cnt[cnt == n].index
    return q.set_index(KEY).loc[keep].reset_index()


def abs_err(p: pd.DataFrame, by: list[str]) -> pd.Series:
    """Per-row absolute error, averaged over seeds."""
    return p.assign(ae=(p.y_true - p.y_pred).abs()).groupby(by + KEY).ae.mean()


def comparison(p: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for k, g in p.groupby(by + ["seed"]):
        rows.append({**dict(zip(by + ["seed"], k, strict=True)), **metrics(g.y_true, g.y_pred)})
    t = pd.DataFrame(rows)
    cols = ["rmse", "mae", "mape", "r2", "rmse_pct", "bias"]
    agg = t.groupby(by)[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_rows"] = t.groupby(by).n.first()
    agg["n_seeds"] = t.groupby(by).seed.nunique()
    return agg.reset_index()


def report(run_dirs: list[Path], out: Path, input_dir: Path) -> None:
    cfg = load_config()
    out.mkdir(parents=True, exist_ok=True)
    P = load_preds(run_dirs)
    written = []

    def save(df: pd.DataFrame, name: str) -> None:
        df.to_csv(out / name, index=False)
        written.append(name)

    # ---- 1. forward chaining: same rows for every mask x lead x model x seed
    fw = P[P.kind == "forward"]
    if len(fw):
        fc = common(fw, ["mask", "lead", "model", "seed"])
        save(comparison(fc, ["mask", "lead", "model"]), "masked_model_comparison.csv")
        ae = abs_err(fc, ["mask", "lead", "model"])
        rows = []
        for (lead, model), _ in fc[fc.model != "baseline_last5"].groupby(["lead", "model"]):
            d = ae.loc[("unmasked", lead, model)] - ae.loc[("masked", lead, model)]
            rows.append({"lead": lead, "model": model, "comparison": "masked vs unmasked",
                         **paired_test(d)})
        save(_with_holm(rows, "masked vs unmasked, forward, models x leads"),
             "masked_vs_unmasked_wilcoxon.csv")
        rows = []
        for (mask, lead, model), _ in fc[fc.model != "baseline_last5"].groupby(
                ["mask", "lead", "model"]):
            d = ae.loc[(mask, lead, "baseline_last5")] - ae.loc[(mask, lead, model)]
            rows.append({"mask": mask, "lead": lead, "model": model,
                         "comparison": "model vs baseline_last5", **paired_test(d)})
        save(_with_holm(rows, "model vs baseline, forward, masks x leads x models"),
             "masked_wilcoxon_vs_baseline.csv")
        # earliness: earlier lead vs lead 4 (d > 0: the earlier forecast has smaller error)
        leads = sorted(fc.lead.unique())
        rows = []
        if 4 in leads:
            for (mask, model), _ in fc.groupby(["mask", "model"]):
                for L in [x for x in leads if x != 4]:
                    d = ae.loc[(mask, 4, model)] - ae.loc[(mask, L, model)]
                    rows.append({"mask": mask, "model": model, "lead": L,
                                 "comparison": f"lead {L} vs lead 4", **paired_test(d)})
            save(_with_holm(rows, "earlier lead vs lead 4, masks x models x leads"),
                 "masked_earliness_wilcoxon.csv")

    # ---- 2. leave-district-out
    ld = P[P.kind == "ldo"]
    if len(ld):
        lc = common(ld, ["mask", "lead", "model"])
        save(comparison(lc, ["mask", "lead", "model"]), "masked_leave_district_out.csv")
        ae = abs_err(lc, ["mask", "lead", "model"])
        rows = []
        for (lead, model), _ in lc[lc.model != "baseline_last5"].groupby(["lead", "model"]):
            d = ae.loc[("unmasked", lead, model)] - ae.loc[("masked", lead, model)]
            rows.append({"lead": lead, "model": model, "comparison": "masked vs unmasked (LDO)",
                         **paired_test(d)})
        save(_with_holm(rows, "masked vs unmasked, leave-district-out, models x leads"),
             "masked_vs_unmasked_ldo_wilcoxon.csv")

    # ---- 3. conformal (Mondrian by season and standard), nominal 1 - alpha
    cf = P[P.kind == "conformal"]
    if len(cf):
        cc = common(cf, ["mask", "lead", "model"])
        rows = []
        for v in ("standard", "mondrian_season"):
            for a in sorted(cfg["conformal"]["reliability_alphas"]):
                q = cc[f"q_{v}_{a}"]
                d = cc.assign(variant=v, alpha=a, covered=(cc.y_true - cc.y_pred).abs() <= q,
                              width=2 * q, infinite=~np.isfinite(q))
                rows.append(d)
        D = pd.concat(rows, ignore_index=True)
        fin = D.assign(width=D.width.where(~D.infinite))
        for by, name in ((["mask", "lead", "model", "variant", "alpha"], "masked_conformal.csv"),
                         (["mask", "lead", "model", "variant", "alpha", "season"],
                          "masked_conformal_by_season.csv")):
            g = fin.groupby(by)
            t = g.agg(n=("covered", "size"), coverage=("covered", "mean"),
                      mean_width_finite=("width", "mean"), share_infinite=("infinite", "mean"),
                      mean_y=("y_true", "mean")).reset_index()
            t["nominal"] = 1 - t.alpha
            t["width_pct_of_mean"] = 100 * t.mean_width_finite / t.mean_y
            save(t, name)
        save(cc.groupby(["mask", "lead", "model"]).apply(
            lambda g: pd.Series(metrics(g.y_true, g.y_pred)), include_groups=False)
            .reset_index(), "masked_conformal_point_metrics.csv")
        cc["n_calib_same_season"] = np.where(cc.season == "Maha", cc.n_calib_Maha,
                                             cc.n_calib_Yala)
        save(cc.groupby(["mask", "lead", "season"])[["n_calib", "n_calib_same_season"]]
             .agg(["min", "median"]).pipe(lambda t: t.set_axis(
                 [f"{a}_{b}" for a, b in t.columns], axis=1)).reset_index(),
             "masked_conformal_calibration_sizes.csv")

    # ---- 4. ablation (RF) + SHAP
    ab = P[P.kind == "abl"]
    if len(ab):
        rows_t, rows_w = [], []
        for (mask, lead), g in ab.groupby(["mask", "lead"]):
            gc = common(g, ["variant"])
            for v, h in gc.groupby("variant"):
                rows_t.append({"mask": mask, "lead": lead, "variant": v,
                               **metrics(h.y_true, h.y_pred)})
            e = abs_err(gc, ["variant"])
            w = []
            for v in sorted(set(gc.variant) - {"baseline"}):
                for ref in ("baseline", "all"):
                    if v == ref:
                        continue
                    w.append({"mask": mask, "lead": lead, "variant": v, "vs": ref,
                              **paired_test(e.loc[ref] - e.loc[v])})
            rows_w.append(_with_holm(w, f"ablation {mask} L{lead}"))
        save(pd.DataFrame(rows_t).sort_values(["mask", "lead", "rmse"]), "masked_ablation.csv")
        save(pd.concat(rows_w, ignore_index=True), "masked_ablation_wilcoxon.csv")
    shp = []
    for d in run_dirs:
        for f in sorted((d / "shap").glob("shap_*_L*.parquet")):
            m = re.match(r"shap_(masked|unmasked)_L(\d+)", f.stem)
            sv = pd.read_parquet(f)
            fcols = [c for c in sv.columns if c not in KEY + ["fold"]]
            imp = sv[fcols].abs().mean().rename("mean_abs_shap").reset_index()
            shp.append(imp.rename(columns={"index": "feature"})
                       .assign(mask=m[1], lead=int(m[2])))
    if shp:
        S = pd.concat(shp, ignore_index=True)
        fl = input_dir / "feature_list.json"
        if fl.exists():
            groups = json.loads(fl.read_text(encoding="utf-8"))["groups"]
            gof = {c: g for g, cs in groups.items() for c in cs} | {"is_maha": "season flag"}
            S["group"] = S.feature.map(gof)
        save(S.sort_values(["mask", "lead", "mean_abs_shap"], ascending=[True, True, False]),
             "masked_shap_feature_importance.csv")
        if "group" in S:
            G = S.groupby(["mask", "lead", "group"]).mean_abs_shap.sum().rename("sum")
            G = (G / G.groupby(["mask", "lead"]).transform("sum")).rename("share").reset_index()
            save(G, "masked_shap_group_share.csv")
    print("wrote:", ", ".join(written))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["models", "followups", "report"])
    ap.add_argument("--input", default="data/processed")
    ap.add_argument("--out", default="results/runs/06_masked")
    ap.add_argument("--runs", nargs="+", help="report: run folders with preds/ and shap/")
    ap.add_argument("--tables", default="results/tables", help="report: output folder")
    a = ap.parse_args()
    if a.stage == "report":
        report([Path(r) for r in (a.runs or [a.out])], Path(a.tables), Path(a.input))
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    {"models": stage_models, "followups": stage_followups}[a.stage](Path(a.input), out)


if __name__ == "__main__":
    main()
