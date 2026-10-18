"""Shock-season evaluation from saved forward-chaining predictions (no training).

    python -m src.analysis.shocks --pred results/tables/predictions_models.csv

Shock seasons: (1) named in advance in configs/config.yaml (policy/economic 2021-22, the
2010/11 Maha floods) and (2) the k largest other negative national yield anomalies, chosen from
the DCS data (cause not verified). Every metric is computed from the prediction file given.
Outputs: results/tables/shock_seasons.{csv,md}, results/figures/shock_bias_by_season.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.plotstyle import GRID, INK, MODEL_LABEL, MUTED, SURFACE, line_kw, plt, style
from src.analysis.stats import to_markdown
from src.config import REPO_ROOT, load_config
from src.data.build_master import shock_candidates

TABLES, FIGS = REPO_ROOT / "results" / "tables", REPO_ROOT / "results" / "figures"


def shock_list(cfg: dict, labels: pd.DataFrame) -> pd.DataFrame:
    sc = cfg["shocks"]
    named = [(s, y, "policy/economic (named)") for s, y in sc["policy_economic"]]
    named += [(s, y, "weather (named)") for s, y in sc["weather_named"]]
    first = cfg["data"]["modelling_first_season"]
    cand = shock_candidates(labels, first, k=20)
    taken = {(s, y) for s, y, _ in named}
    extra = cand[~cand.apply(lambda r: (r.season, r.season_year) in taken, axis=1)]
    extra = extra[extra.anom_pct < 0].nsmallest(sc["data_chosen_k"], "anom_pct")
    rows = named + [(r.season, int(r.season_year), "data-chosen (cause unverified)")
                    for r in extra.itertuples()]
    out = pd.DataFrame(rows, columns=["season", "year", "shock_type"])
    return out.merge(cand.rename(columns={"season_year": "year"})[
        ["season", "year", "nat_yield", "past10_mean", "anom_pct"]], on=["season", "year"],
        how="left")


def seed_mean(pred: pd.DataFrame) -> pd.DataFrame:
    return (pred.groupby(["model", "district", "season", "year"], as_index=False)
            .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean")))


def _m(g: pd.DataFrame) -> pd.Series:
    e = g.y_pred - g.y_true
    return pd.Series({"n": len(g), "rmse": float(np.sqrt(np.mean(e ** 2))),
                      "mae": float(np.mean(np.abs(e))), "bias": float(np.mean(e)),
                      "mean_true": float(g.y_true.mean())})


def shock_table(p: pd.DataFrame, shocks: pd.DataFrame) -> pd.DataFrame:
    p = p.merge(shocks[["season", "year", "shock_type"]], on=["season", "year"], how="left")
    p["shock_type"] = p.shock_type.fillna("other test seasons")
    sh = p[p.shock_type != "other test seasons"]
    per = (sh.groupby(["shock_type", "season", "year", "model"])
           .apply(_m, include_groups=False).reset_index())
    pooled = p.groupby(["shock_type", "model"]).apply(_m, include_groups=False).reset_index()
    pooled["season"], pooled["year"] = "all", -1
    return pd.concat([per, pooled], ignore_index=True)


def season_bias_figure(p: pd.DataFrame, shocks: pd.DataFrame, models: list[str], out: Path,
                       dpi: int) -> None:
    p = p.copy()
    p["harvest_year"] = np.where(p.season == "Maha", p.year + 1, p.year)
    p["order"] = 2 * p.harvest_year + (p.season == "Yala")
    s = (p.groupby(["model", "order", "season", "year"]).apply(
        lambda g: float(np.mean(g.y_pred - g.y_true)), include_groups=False)
        .rename("bias").reset_index())
    orders = sorted(s.order.unique())
    lab = s.drop_duplicates("order").set_index("order")
    fig, ax = plt.subplots(figsize=(12, 4.4), facecolor=SURFACE)
    style(ax)
    sh = shocks.assign(harvest_year=np.where(shocks.season == "Maha", shocks.year + 1, shocks.year))
    for o in 2 * sh.harvest_year + (sh.season == "Yala"):
        if o in orders:
            i = orders.index(o)
            ax.axvspan(i - 0.45, i + 0.45, color=GRID, linewidth=0, zorder=0)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    for m in models:
        g = s[s.model == m].set_index("order").reindex(orders)
        ax.plot(range(len(orders)), g.bias, label=MODEL_LABEL.get(m, m), marker="o",
                markersize=3.5, **line_kw(m))
    ticks = [f"{'M' if r.season == 'Maha' else 'Y'} {r.year}" for r in lab.loc[orders].itertuples()]
    ax.set_xticks(range(len(orders)), ticks, rotation=90)
    ax.set_xlim(-0.7, len(orders) - 0.3)
    ax.set_ylabel("mean error, predicted − observed (kg/ha)", color=MUTED, fontsize=9)
    ax.set_title("Mean forecast error per test season (shaded = shock seasons; above 0 = "
                 "over-forecast)", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=len(models))
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def main(pred_path: Path, tag: str = "") -> None:
    cfg = load_config()
    labels = pd.read_csv(REPO_ROOT / "data" / "interim" / "dcs_yield_tidy.csv")
    shocks = shock_list(cfg, labels)
    pred = pd.read_csv(pred_path)
    p = seed_mean(pred)
    models = [m for m in ["baseline_last5", "random_forest", "ridge", "lightgbm", "xgboost",
                          "hybrid_lgbm"] if m in set(p.model)]
    t = shock_table(p, shocks)
    t["model"] = pd.Categorical(t.model, models)
    t = t.sort_values(["shock_type", "year", "season", "model"])
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABLES / f"shock_seasons{tag}.csv", index=False)
    shocks.to_csv(TABLES / f"shock_season_list{tag}.csv", index=False)
    md = t.copy()
    md["season"] = md.season.where(md.year > 0, "pooled")
    md["year"] = md.year.where(md.year > 0, "")
    (TABLES / f"shock_seasons{tag}.md").write_text(
        to_markdown(md[["shock_type", "season", "year", "model", "n", "rmse", "mae", "bias",
                        "mean_true"]]) + "\n", encoding="utf-8")
    season_bias_figure(p, shocks, [m for m in ["baseline_last5", "random_forest", "ridge"]
                                   if m in models], FIGS / f"shock_bias_by_season{tag}.png",
                       cfg["figures"]["dpi"])
    print(shocks.to_string())
    print(t[t.year == -1].round(1).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(TABLES / "predictions_models.csv"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main(Path(a.pred), a.tag)
