"""Step C3: master table (one row per district-season with a yield, 2000/01 onward).

    python -m src.data.build_master                 # lead 0, nominal windows
    python -m src.data.build_master --lead 6        # earliness variant
    python -m src.data.build_master --shift -2      # window-shift sensitivity variant

Inputs : data/interim/dcs_yield_tidy.csv, data/interim/ee/{modis,chirps}_YYYY.csv,
         data/interim/ee/static_cropland.csv
Outputs: data/processed/master_table.parquet (or variants/master_lead{L}_shift{S}.parquet)
         data/processed/feature_list.json, data/processed/series_{rain,modis}.parquet,
         docs/data_card.md (nominal build only)
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd

from src.config import REPO_ROOT, load_config
from src.data.features import (
    Calendar,
    build_season_features,
    chirps_daily,
    era5_daily,
    modis_clean,
)

EE = REPO_ROOT / "data" / "interim" / "ee"
PROC = REPO_ROOT / "data" / "processed"

GROUPS = {
    "lagged_yield": ("y_", "sown_prev_same", "harv_ratio_prev_same", "n_label_hist_same"),
    "rainfall": ("rain_",),
    "modis": ("ndvi_", "evi_", "modis_valid_"),
    "temp_radiation": ("tmean_", "tmax_", "tmin_", "vpd_", "srad_", "evap_", "heat_days"),
}


def load_series(pattern: str) -> pd.DataFrame | None:
    files = sorted(glob.glob(str(EE / pattern)))
    if not files:
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def feature_groups(cols: list[str]) -> dict[str, list[str]]:
    out = {g: [c for c in cols if c.startswith(p)] for g, p in GROUPS.items()}
    return {g: v for g, v in out.items() if v}


def shock_candidates(labels: pd.DataFrame, first: dict, k: int = 5) -> pd.DataFrame:
    """National area-weighted yield anomaly vs the mean of the previous 10 same seasons."""
    d = labels.dropna(subset=["yield_kg_ha", "harvested_ha"])
    nat = (d.groupby(["season", "season_year"])
           .apply(lambda g: np.average(g.yield_kg_ha, weights=g.harvested_ha),
                  include_groups=False).rename("nat_yield").reset_index())
    nat["past10_mean"] = nat.groupby("season").nat_yield.transform(
        lambda s: s.shift(1).rolling(10, min_periods=5).mean())
    nat["anom_pct"] = (nat.nat_yield / nat.past10_mean - 1) * 100
    nat = nat[nat.season_year >= nat.season.map(first)]
    return nat.sort_values("anom_pct").groupby("season").head(k).reset_index(drop=True)


def main(lead: int, shift: int) -> None:
    cfg = load_config()
    cal = Calendar.from_cfg(cfg)
    labels = pd.read_csv(REPO_ROOT / "data" / "interim" / "dcs_yield_tidy.csv")
    pcodes = labels.drop_duplicates("district")[["district", "adm2_pcode"]]

    chirps_raw, modis_raw = load_series("chirps_*.csv"), load_series("modis_*.csv")
    era5_raw = load_series("era5_*.csv")
    era5 = era5_daily(era5_raw) if era5_raw is not None else None
    sp = EE / "static_cropland.csv"
    static = pd.read_csv(sp) if sp.exists() else None
    if chirps_raw is None and modis_raw is None:
        raise SystemExit("No Earth Engine CSVs in data/interim/ee/ - run src.data.ee_extract")
    rain = chirps_daily(chirps_raw) if chirps_raw is not None else None
    modis = modis_clean(modis_raw, static) if modis_raw is not None else None

    first_year = int(chirps_raw.start.str[:4].astype(int).min()) if chirps_raw is not None else 2000
    last_year = int(labels.season_year.max())
    grid = pd.DataFrame([(d, p, s, y) for d, p in pcodes.itertuples(index=False)
                         for s in ("Maha", "Yala") for y in range(first_year, last_year + 1)],
                        columns=["district", "adm2_pcode", "season", "season_year"])
    feats = build_season_features(grid, labels, rain, modis, cal, lead, shift, era5=era5,
                                  heat_thr=cfg["features"]["heat_day_tmax_c"])

    first = cfg["data"]["modelling_first_season"]
    keep = ["district", "adm2_pcode", "season", "season_label", "season_year", "harvest_year",
            "season_order", "yield_kg_ha", "mahaweli_merged", "sanity_flag"]
    m = labels[keep].merge(feats, on=["district", "season", "season_year"], how="left")
    m = m[m.season_year >= m.season.map(first)].reset_index(drop=True)
    m["mahaweli_affected"] = m.district.isin(cfg["data"]["mahaweli_affected_districts"]).astype(int)
    m["diag_post_merge_affected"] = m.mahaweli_affected * m.mahaweli_merged

    feat_cols = [c for c in feats.columns
                 if c not in ("district", "season", "season_year", "forecast_date",
                              "lead_weeks", "shift_weeks")]
    forbidden = set(cfg["forbidden_features"])
    assert not forbidden & set(feat_cols), forbidden & set(feat_cols)
    groups = feature_groups(feat_cols)

    PROC.mkdir(parents=True, exist_ok=True)
    nominal = lead == 0 and shift == 0
    out = (PROC / "master_table.parquet" if nominal
           else PROC / "variants" / f"master_lead{lead}_shift{shift}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out, index=False)
    print(f"wrote {out.relative_to(REPO_ROOT)}: {m.shape[0]} rows x {m.shape[1]} cols")
    if not nominal:
        return

    (PROC / "feature_list.json").write_text(json.dumps(
        {"target": "yield_kg_ha", "groups": groups, "forbidden": sorted(forbidden),
         "diagnostic_only": ["mahaweli_merged", "mahaweli_affected", "diag_post_merge_affected",
                             "sanity_flag"]}, indent=2), encoding="utf-8")
    if rain is not None:
        rain.to_parquet(PROC / "series_rain_daily.parquet", index=False)
    if modis is not None:
        modis.to_parquet(PROC / "series_modis.parquet", index=False)
    if era5 is not None:
        era5.to_parquet(PROC / "series_era5_daily.parquet", index=False)
    write_data_card(m, groups, labels, first, cfg)


def write_data_card(m, groups, labels, first, cfg) -> None:
    cov = []
    for g, cols in groups.items():
        raw_cols = [c for c in cols if not c.endswith("_anom")]
        nn = m[raw_cols].notna().mean()
        cov.append((g, len(cols), f"{nn.mean() * 100:.1f}", f"{nn.min() * 100:.1f}"))
    shocks = shock_candidates(labels, first)
    by_season = m.groupby("season").agg(rows=("yield_kg_ha", "size"),
                                        first=("season_year", "min"), last=("season_year", "max"),
                                        districts=("district", "nunique"))
    lines = [
        "# Data card: master_table.parquet", "",
        f"Built by `src/data/build_master.py` (lead 0, nominal windows). Target: `yield_kg_ha`. "
        f"**{len(m)} rows**, {m.district.nunique()} districts.", "",
        "**Scope:** the modelling table starts where satellite and climate features exist "
        f"(Maha {first['Maha']}/{str(first['Maha'] + 1)[-2:]}, Yala {first['Yala']}). "
        "DCS yields from 1979–1999 are used ONLY to compute lagged-yield features and the "
        "baseline; they are never training rows.", "",
        "| season | rows | first season_year | last season_year | districts |",
        "|---|---|---|---|---|",
        *[f"| {s} | {r.rows} | {r.first} | {r.last} | {r.districts} |"
          for s, r in by_season.iterrows()],
        "", "## Feature coverage (non-missing share of rows)", "",
        "| group | n features (incl. anomalies) | mean coverage % | worst feature % |",
        "|---|---|---|---|",
        *[f"| {g} | {n} | {a} | {w} |" for g, n, a, w in cov],
        "", "Anomalies are relative to the same district-season in PREVIOUS years only "
        f"(at least {cfg['features']['anomaly_min_history_years']} years). Yield lags respect the "
        f"assumed {cfg['data']['label_availability_lag_months']}-month publication lag.", "",
        "## Diagnostic-only columns (never features)", "",
        "`mahaweli_merged`, `mahaweli_affected`, `diag_post_merge_affected`, `sanity_flag` "
        f"({int(m.sanity_flag.sum())} rows flagged).", "",
        "## Shock-season candidates (from the data)", "",
        "Largest negative national area-weighted yield anomalies vs the previous 10 same "
        "seasons. 2010/11 Maha (January–February 2011 floods) was named in advance.", "",
        "| season | season_year | national yield kg/ha | past-10 mean | anomaly % |",
        "|---|---|---|---|---|",
        *[f"| {r.season} | {r.season_year} | {r.nat_yield:.0f} | {r.past10_mean:.0f} | "
          f"{r.anom_pct:.1f} |" for r in shocks.itertuples()],
        "", "## Known limitations", "",
        "See `results/EXPLAINED.md` → Limitations observed, and `docs/decisions.md`.",
    ]
    (REPO_ROOT / "docs" / "data_card.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote docs/data_card.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=int, default=0)
    ap.add_argument("--shift", type=int, default=0)
    a = ap.parse_args()
    main(a.lead, a.shift)
