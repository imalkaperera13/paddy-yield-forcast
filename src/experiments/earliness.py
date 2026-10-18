"""Experiment group 3: earliness (lead time) and growth-window shift sensitivity.

    python -m src.experiments.earliness --input data/processed --out results/runs/03_earliness

Each lead L uses its own feature table (variants/master_lead{L}_shift0.parquet; lead 0 is
master_table.parquet), built so that only data available L weeks before the reference harvest
enters, and its own forward-chaining folds (labels published by THAT forecast date).
Shift S uses windows moved S weeks (variants/master_lead0_shift{S}.parquet).
Every lead/shift is scored on the same district-seasons.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from src.analysis.plotstyle import INK, MODEL_LABEL, MUTED, SURFACE, line_kw, plt, style
from src.analysis.stats import to_markdown
from src.config import load_config, set_seed
from src.experiments.models import load_inputs
from src.models.metrics import metrics_table
from src.models.train import run_model

KEYS = ["district", "season", "season_year"]


def table_path(input_dir: Path, lead: int, shift: int) -> Path:
    if lead == 0 and shift == 0:
        return input_dir / "master_table.parquet"
    name = f"master_lead{lead}_shift{shift}.parquet"
    nested = input_dir / "variants" / name
    return nested if nested.exists() else input_dir / name


def same_rows(pred: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    p = pred[pred.y_pred.notna() & pred.baseline_pred.notna()]
    n = p.groupby(group).ngroups
    cnt = p.groupby(KEYS).size()
    keep = cnt[cnt == n].index
    return p.set_index(KEYS).loc[keep].reset_index()


def wilcoxon_rows(p: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    base = p[p.model == "baseline_last5"]
    for (v, m), g in p[p.model != "baseline_last5"].groupby([by, "model"]):
        b = base[base[by] == v].set_index(KEYS).loc[g.set_index(KEYS).index]
        em = (g.set_index(KEYS).y_pred - g.set_index(KEYS).y_true).abs()
        eb = (b.y_pred - b.y_true).abs()
        d = (eb - em).to_numpy()
        d = d[d != 0]
        w = wilcoxon(d)
        rpos = pd.Series(abs(d)).rank()
        r = float((rpos[d > 0].sum() - rpos[d < 0].sum()) / rpos.sum())
        rows.append({by: v, "model": m, "n_pairs": len(d), "p_value": float(w.pvalue),
                     "rank_biserial_r": r, "share_rows_better": float((d > 0).mean())})
    t = pd.DataFrame(rows)
    # Holm across all comparisons in this table
    order = t.p_value.sort_values().index
    k = len(t)
    adj, run = pd.Series(index=t.index, dtype=float), 0.0
    for i, ix in enumerate(order):
        run = max(run, min(1.0, (k - i) * t.p_value[ix]))
        adj[ix] = run
    t["p_holm"] = adj
    return t


def lead_figure(t: pd.DataFrame, out: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)
    for ax, col, lab in zip(axes, ["rmse", "r2"], ["RMSE (kg/ha)", "R²"], strict=True):
        style(ax)
        for m in t.model.unique():
            g = t[t.model == m].sort_values("lead_weeks")
            ax.plot(g.lead_weeks, g[col], marker="o", markersize=4, label=MODEL_LABEL.get(m, m),
                    **line_kw(m))
        ax.invert_xaxis()
        ax.set_xlabel("weeks before reference harvest", color=MUTED, fontsize=9)
        ax.set_title(lab, loc="left", fontsize=10, color=INK)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def main(input_dir: Path, out_dir: Path) -> None:
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    seed = cfg["followup_seed"]
    models = (os.environ.get("PADDY_MODELS") or ",".join(cfg["followup_models"])).split(",")
    leads = cfg["earliness"]["lead_weeks_before_harvest"]
    shifts = cfg["features"]["window_shift_sensitivity_weeks"]
    tables, figs = out_dir / "tables", out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    _, features = load_inputs(input_dir, cfg)   # same feature list for every variant

    runs = [(L, 0) for L in sorted(leads)] + [(0, s) for s in shifts]
    preds = []
    for L, s in runs:
        df = pd.read_parquet(table_path(input_dir, L, s))
        missing = set(features) - set(df.columns)
        if missing:
            raise RuntimeError(f"lead {L} shift {s}: missing features {sorted(missing)[:5]}")
        for m in models:
            p = run_model(df, features, m, seed, cfg, out_dir, lead=L, tag=f"lead{L}_shift{s}")
            preds.append(p.assign(shift_weeks=s))
    pred = pd.concat(preds, ignore_index=True)
    pred["variant"] = "lead" + pred.lead_weeks.astype(str) + "_shift" + pred.shift_weeks.astype(str)

    # earliness: same rows across all leads and models
    pe = same_rows(pred[pred.shift_weeks == 0], ["variant", "model"])
    te = metrics_table(pe, ["lead_weeks", "model"])
    te.to_csv(tables / "earliness.csv", index=False)
    (tables / "earliness.md").write_text(
        to_markdown(te[["lead_weeks", "model", "n", "rmse", "mae", "r2", "bias"]]) + "\n",
        encoding="utf-8")
    we = wilcoxon_rows(pe, "lead_weeks")
    we.to_csv(tables / "earliness_wilcoxon_vs_baseline.csv", index=False)
    (tables / "earliness_wilcoxon_vs_baseline.md").write_text(to_markdown(we, ".4g") + "\n",
                                                             encoding="utf-8")
    lead_figure(te, figs / "earliness_curve.png", cfg["figures"]["dpi"])

    # window shift: lead 0, shifts -2/0/+2, same rows
    ps = same_rows(pred[pred.lead_weeks == 0], ["variant", "model"])
    ts = metrics_table(ps, ["shift_weeks", "model"])
    ts.to_csv(tables / "window_shift_sensitivity.csv", index=False)
    (tables / "window_shift_sensitivity.md").write_text(
        to_markdown(ts[["shift_weeks", "model", "n", "rmse", "mae", "r2"]]) + "\n",
        encoding="utf-8")
    cols = KEYS + ["y_true", "y_pred", "model", "fold", "seed", "lead_weeks", "shift_weeks"]
    pred[cols].rename(columns={"season_year": "year"}).to_csv(
        tables / "predictions_earliness.csv", index=False)
    print(te[["lead_weeks", "model", "n", "rmse", "r2"]].round(3).to_string())
    print(ts[["shift_weeks", "model", "n", "rmse", "r2"]].round(3).to_string())
    print("experiment 03 complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed")
    ap.add_argument("--out", default="results/runs/03_earliness")
    a = ap.parse_args()
    main(Path(a.input), Path(a.out))
