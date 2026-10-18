"""Report figures and the combined predictions file, from saved outputs only (no training).

    python -m src.analysis.report_figures

Outputs
  results/figures/validation_scheme.png   forward-chaining folds (train / test / gap)
  results/figures/district_error_map.png  per-district RMSE (random forest) and RF - baseline
  results/tables/district_errors.csv
  results/predictions.csv                 kernel-02 conformal predictions with lo/hi (α = 0.1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from src.analysis.plotstyle import GRID, INK, MUTED, SLOTS, SURFACE, plt, style
from src.config import REPO_ROOT, load_config

TABLES, FIGS = REPO_ROOT / "results" / "tables", REPO_ROOT / "results" / "figures"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95"])
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec8f8e",
                                                "#a82e2d"])


def validation_scheme(dpi: int) -> None:
    f = pd.read_csv(TABLES / "validation_folds_lead0.csv")
    f = f[f.held_out_district.isna()].reset_index(drop=True)
    yrs_lo, yrs_hi = int(f.train_first_harvest.min()), int(f.test_season_year.max()) + 2
    fig, ax = plt.subplots(figsize=(10, 6.4), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    for i, r in f.iterrows():
        test_h = r.test_season_year + (1 if r.test_season == "Maha" else 0)
        test_x = test_h + (0.0 if r.test_season == "Maha" else 0.5)
        ax.barh(i, r.train_last_harvest + 1 - r.train_first_harvest, left=r.train_first_harvest,
                height=0.62, color=SLOTS[0], linewidth=0)
        ax.barh(i, 0.45, left=test_x, height=0.62, color=SLOTS[1], linewidth=0)
    ax.set_yticks(range(len(f)), [f"{'M' if s == 'Maha' else 'Y'} {y}" for s, y in
                                  zip(f.test_season, f.test_season_year, strict=True)], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlim(yrs_lo - 0.5, yrs_hi)
    ax.set_xlabel("harvest year", color=MUTED, fontsize=9)
    ax.set_ylabel("test season (one fold each)", color=MUTED, fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SLOTS[0], label="training: labels published by the forecast "
                             "date"), Patch(color=SLOTS[1], label="test season")],
              frameon=False, fontsize=8, loc="upper right")
    ax.set_title(f"Forward-chaining validation: {len(f)} folds, expanding window, no future "
                 "data", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "validation_scheme.png", dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def district_errors() -> pd.DataFrame:
    p = pd.read_csv(TABLES / "predictions_models.csv")
    p = (p.groupby(["model", "district", "season", "year"], as_index=False)
         .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean")))
    p["se"] = (p.y_pred - p.y_true) ** 2
    t = p.groupby(["district", "model"]).agg(n=("se", "size"), rmse=("se", "mean")).reset_index()
    t["rmse"] = np.sqrt(t.rmse)
    w = t.pivot(index="district", columns="model", values="rmse")
    w["n"] = t.groupby("district").n.first()
    w["rf_minus_baseline"] = w.random_forest - w.baseline_last5
    return w.reset_index()


def district_map(w: pd.DataFrame, dpi: int) -> None:
    import geopandas as gpd
    g = gpd.read_file(REPO_ROOT / "data" / "interim" / "districts_adm2_simplified.geojson")
    dm = pd.read_csv(REPO_ROOT / "configs" / "district_map.csv")
    g = g.merge(dm[["adm2_pcode", "district"]], on="adm2_pcode").merge(w, on="district")
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.6), facecolor=SURFACE)
    lim = float(np.nanmax(np.abs(g.rf_minus_baseline)))
    from matplotlib.colors import Normalize
    rf_norm = Normalize(float(g.random_forest.min()), float(g.random_forest.max()))
    panels = [("random_forest", SEQ, rf_norm, "Random forest RMSE per district (kg/ha)"),
              ("rf_minus_baseline", DIV, TwoSlopeNorm(0, -lim, lim),
               "RF − baseline RMSE (kg/ha); blue = RF better")]
    for ax, (col, cmap, norm, title) in zip(axes, panels, strict=True):
        ax.set_facecolor(SURFACE)
        g.plot(color=[cmap(norm(v)) for v in g[col]], ax=ax, edgecolor=SURFACE, linewidth=0.8)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, shrink=0.6)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, colors=MUTED, length=0)
        ax.set_axis_off()
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    fig.text(0.01, 0.01, "Forward-chaining test seasons 2009/10–2025/26, seed-averaged "
             "(kernel 01 v2). Jaffna: Maha only.", fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIGS / "district_error_map.png", dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def predictions_file() -> pd.DataFrame:
    p = pd.read_csv(TABLES / "predictions_conformal.csv")
    cols = ["district", "season", "year", "y_true", "y_pred", "lo", "hi", "model", "fold", "seed"]
    out = p[cols].copy()
    out.to_csv(REPO_ROOT / "results" / "predictions.csv", index=False)
    return out


def main() -> None:
    cfg = load_config()
    dpi = cfg["figures"]["dpi"]
    FIGS.mkdir(parents=True, exist_ok=True)
    validation_scheme(dpi)
    w = district_errors()
    w.to_csv(TABLES / "district_errors.csv", index=False)
    district_map(w, dpi)
    p = predictions_file()
    print(w[["district", "n", "baseline_last5", "random_forest", "rf_minus_baseline"]]
          .round(1).sort_values("rf_minus_baseline").to_string(index=False))
    print(f"predictions.csv: {len(p)} rows")


if __name__ == "__main__":
    main()
