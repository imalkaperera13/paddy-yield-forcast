"""Step 2 check: active paddy mask area vs DCS sown extent, per season (no Earth Engine).

Run after `python -m src.data.paddy_mask`:

    .venv/Scripts/python -m scripts.mask_validation

DCS sown extent is published per season, so each Maha and each Yala mask is compared with
that season's sown_ha (never with the annual Maha + Yala total).

Outputs
- results/tables/mask_area_by_season.csv   one row per district x season x year
- results/tables/mask_validation.csv       agreement per district x season, per season
                                            (all districts) and overall, for both variants
Metrics (n = seasons with both values)
- pearson_r, spearman_r : year-to-year agreement in area
- ratio_median          : median of mask_ha / dcs_sown_ha (1 = same area)
- bias_pct              : 100 * (sum mask - sum DCS) / sum DCS
- mape_pct              : mean of |mask - DCS| / DCS * 100
- mae_ha                : mean |mask - DCS| in hectares
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from src.config import REPO_ROOT, load_config

MASK_DIR = REPO_ROOT / "data" / "interim" / "ee" / "mask"
DCS = REPO_ROOT / "data" / "interim" / "dcs_yield_tidy.csv"
DETAIL = REPO_ROOT / "results" / "tables" / "mask_area_by_season.csv"
SUMMARY = REPO_ROOT / "results" / "tables" / "mask_validation.csv"
VARIANTS = {"strict": "mask_strict_ha", "relaxed": "mask_relaxed_ha"}


def agreement(mask: pd.Series, dcs: pd.Series) -> dict:
    ok = mask.notna() & dcs.notna() & (dcs > 0)
    m, d = mask[ok].astype(float), dcs[ok].astype(float)
    n = int(ok.sum())
    if n == 0:
        return {"n": 0}
    out = {
        "n": n,
        "ratio_median": float(np.median(m / d)),
        "bias_pct": float(100 * (m.sum() - d.sum()) / d.sum()),
        "mape_pct": float(np.mean(np.abs(m - d) / d) * 100),
        "mae_ha": float(np.mean(np.abs(m - d))),
    }
    if n >= 3 and m.std() > 0 and d.std() > 0:
        out["pearson_r"] = float(np.corrcoef(m, d)[0, 1])
        out["spearman_r"] = float(m.rank().corr(d.rank()))
    return out


def load_detail(cfg: dict) -> pd.DataFrame:
    files = sorted(glob.glob(str(MASK_DIR / "mask_area_*.csv")))
    if not files:
        raise RuntimeError(f"no mask files in {MASK_DIR}; run python -m src.data.paddy_mask")
    m = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = pd.read_csv(DCS)[["district", "adm2_pcode", "season", "season_year",
                          "season_label", "sown_ha", "mahaweli_merged"]]
    df = m.merge(d, on=["adm2_pcode", "season", "season_year"], how="left",
                 validate="one_to_one")
    for v, col in VARIANTS.items():
        df[f"ratio_{v}"] = df[col] / df["sown_ha"]
    return df.sort_values(["district", "season", "season_year"])


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = ([("district_season", (dist, s), g) for (dist, s), g in df.groupby(["district", "season"])]
              + [("season_all_districts", ("ALL", s), g) for s, g in df.groupby("season")]
              + [("overall", ("ALL", "ALL"), df)])
    for level, (dist, s), g in groups:
        for v, col in VARIANTS.items():
            r = {"level": level, "district": dist, "season": s, "variant": v,
                 "years": f"{int(g.season_year.min())}-{int(g.season_year.max())}",
                 "mean_flood_clear_share": float(g["flood_clear_share"].mean())}
            r.update(agreement(g[col], g["sown_ha"]))
            rows.append(r)
    cols = ["level", "district", "season", "variant", "years", "n", "pearson_r", "spearman_r",
            "ratio_median", "bias_pct", "mape_pct", "mae_ha", "mean_flood_clear_share"]
    return pd.DataFrame(rows).reindex(columns=cols)


def main() -> None:
    cfg = load_config()
    df = load_detail(cfg)
    DETAIL.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DETAIL, index=False)
    s = summarise(df)
    s.to_csv(SUMMARY, index=False)
    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))
    print(f"\nwrote {DETAIL.relative_to(REPO_ROOT)} and {SUMMARY.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
