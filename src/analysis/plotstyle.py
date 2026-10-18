"""Shared figure style for report figures (300 dpi PNG, light surface).

Categorical slots follow the validated default palette in a fixed order; the baseline is the
reference and is drawn in neutral ink (dashed), never in a series hue.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402,F401

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MODEL_COLOR = {"random_forest": SLOTS[0], "ridge": SLOTS[1], "lightgbm": SLOTS[2],
               "xgboost": SLOTS[3], "hybrid_lgbm": SLOTS[4]}
MODEL_LABEL = {"baseline_last5": "Baseline (last-5 mean)", "random_forest": "Random forest",
               "ridge": "Ridge", "lightgbm": "LightGBM", "xgboost": "XGBoost",
               "hybrid_lgbm": "Hybrid (baseline + LightGBM)"}


def line_kw(model: str) -> dict:
    if model == "baseline_last5":
        return {"color": MUTED, "linewidth": 2, "linestyle": (0, (4, 3))}
    return {"color": MODEL_COLOR.get(model, SLOTS[5]), "linewidth": 2}


def style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
