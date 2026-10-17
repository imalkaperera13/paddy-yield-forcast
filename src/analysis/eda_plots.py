"""Descriptive figures for the data chapter (no model results).

    python -m src.analysis.eda_plots
Outputs results/figures/yield_trends.png and results/figures/season_windows_example.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import REPO_ROOT, load_config  # noqa: E402
from src.data.features import Calendar  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"
BLUE, ORANGE, LIGHT = "#2a78d6", "#e8730c", "#c9c8c2"
FIG = REPO_ROOT / "results" / "figures"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def yield_trends(labels: pd.DataFrame, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, facecolor=SURFACE)
    for ax, season in zip(axes, ["Maha", "Yala"], strict=True):
        d = labels[labels.season == season]
        for _, g in d.groupby("district"):
            g = g.sort_values("season_year")
            ax.plot(g.season_year, g.yield_kg_ha, color=LIGHT, linewidth=0.7)
        w = d.dropna(subset=["harvested_ha"]).groupby("season_year").apply(
            lambda g: np.average(g.yield_kg_ha, weights=g.harvested_ha), include_groups=False)
        ax.plot(w.index, w.values, color=BLUE, linewidth=2.2)
        ax.axvline(1999.5, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.text(2000, 6900, "modelling rows start", color=MUTED, fontsize=8)
        if season == "Maha":
            ax.axvspan(2020.6, 2022.4, color=ORANGE, alpha=0.12, linewidth=0)
            ax.text(2016.2, 1250, "2021/22 fertiliser\nban seasons", color=MUTED, fontsize=8)
            ax.annotate("2010/11 floods", xy=(2010, w.loc[2010]), xytext=(2003, 2000),
                        color=MUTED, fontsize=8, arrowprops={"arrowstyle": "-", "color": MUTED})
        ax.set_title(f"{season}" + (" (labelled by planting year)" if season == "Maha" else ""),
                     loc="left", fontsize=10, color=INK)
        ax.set_xlim(1978, 2026)
        ax.set_ylim(1000, 7200)
        _style(ax)
    axes[0].set_ylabel("yield (kg per net ha)", color=MUTED, fontsize=9)
    fig.legend(handles=[plt.Line2D([], [], color=BLUE, linewidth=2.2),
                        plt.Line2D([], [], color=LIGHT, linewidth=0.7)],
               labels=["national, area-weighted", "individual districts"], frameon=False,
               loc="lower left", bbox_to_anchor=(0.06, -0.04), ncol=2, fontsize=9)
    fig.suptitle("District paddy yield, 1979–2025 (DCS)", x=0.06, ha="left", fontsize=12, color=INK)
    fig.text(0.06, -0.09, "Source: Department of Census and Statistics, Sri Lanka. National line "
             "weighted by gross harvested extent of the 25 districts.", fontsize=8, color=MUTED)
    fig.savefig(FIG / "yield_trends.png", dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def windows_example(cfg: dict, dpi: int, pcode: str = "LK71", name: str = "Anuradhapura",
                    years=(2015, 2016)) -> None:
    cal = Calendar.from_cfg(cfg)
    proc = REPO_ROOT / "data" / "processed"
    rain = pd.read_parquet(proc / "series_rain_daily.parquet")
    modis = pd.read_parquet(proc / "series_modis.parquet")
    t0 = cal.anchor("Maha", years[0]) - pd.Timedelta(days=40)
    t1 = cal.harvest_ref("Yala", years[1]) + pd.Timedelta(days=30)
    r = rain[(rain.adm2_pcode == pcode) & rain.date.between(t0, t1)]
    wk = r.set_index("date").rate.resample("7D").sum()
    m = modis[(modis.adm2_pcode == pcode) & modis.center.between(t0, t1)]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 5.6), sharex=True, facecolor=SURFACE,
                                  gridspec_kw={"height_ratios": [1, 1.2], "hspace": 0.12})
    for a in (ax, ax2):
        for season, sy in (("Maha", years[0]), ("Yala", years[1])):
            for nm, s, e in cal.blocks(season, sy):
                a.axvspan(s, e, color=BLUE if nm.startswith("b") else LIGHT,
                          alpha=0.07 if nm.startswith("b") else 0.25, linewidth=0)
                if a is ax:
                    a.text(s + (e - s) / 2, 0.97, nm, transform=a.get_xaxis_transform(),
                           ha="center", va="top", fontsize=7, color=MUTED)
            hr = cal.harvest_ref(season, sy)
            a.axvline(hr, color=INK, linewidth=0.9)
            if a is ax:
                label = f"{sy}/{str(sy + 1)[-2:]} Maha" if season == "Maha" else f"{sy} Yala"
                a.text(hr, 1.02, f"harvest ref.\n{label}", transform=a.get_xaxis_transform(),
                       ha="center", fontsize=7, color=INK)
    ax.bar(wk.index, wk.values, width=6, color=LIGHT, align="edge")
    ax2.plot(m.center, m.NDVI_mean, color=BLUE, linewidth=2, marker="o", markersize=3)
    ax.set_ylabel("rain per week (mm)", color=MUTED, fontsize=9)
    ax2.set_ylabel("cropland NDVI", color=MUTED, fontsize=9)
    ax2.set_ylim(0.2, 0.95)
    _style(ax)
    _style(ax2)
    fig.suptitle(f"Growth-stage windows on real data: {name} district", x=0.06, ha="left",
                 fontsize=12, color=INK, y=1.08)
    fig.text(0.06, -0.06, "Shaded: pre-season block (grey) and growth blocks b1–b5 (blue), 4 weeks "
             "each from the nominal planting date. CHIRPS pentads summed to weeks; MODIS MOD13Q1 "
             "NDVI over WorldCover cropland.", fontsize=8, color=MUTED)
    fig.savefig(FIG / "season_windows_example.png", dpi=dpi, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    FIG.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(REPO_ROOT / "data" / "interim" / "dcs_yield_tidy.csv")
    yield_trends(labels, cfg["figures"]["dpi"])
    windows_example(cfg, cfg["figures"]["dpi"])
    print("wrote results/figures/yield_trends.png, results/figures/season_windows_example.png")


if __name__ == "__main__":
    main()
