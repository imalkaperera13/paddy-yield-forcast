"""Step 3c: season features inside the static paddy mask, at forecast cutoffs 12, 8, 4 weeks.

    .venv/Scripts/python -m src.data.build_masked_features            # leads 12 8 4
    .venv/Scripts/python -m src.data.build_masked_features --leads 12

Inputs : data/interim/ee_masked/<season>/<group>/*.csv (from src.data.ee_extract_masked;
         every district group found is used), data/interim/dcs_yield_tidy.csv
Outputs: data/processed/masked/features_lead{L}.parquet, one row per district x season with
         a DCS yield, for the districts that have masked series.

The feature code is the pilot's src/data/features.py, unchanged, so the same cutoff rules
apply: forecast_date = reference harvest date - lead weeks; a MODIS composite, CHIRPS pentad
or ERA5 day counts only if it ends (+ latency) on or before forecast_date; anomalies use
previous years only; yield lags only once published (6-month lag assumption).
Maha rows use Maha-mask series, Yala rows Yala-mask series.
The unmasked pilot features for the same leads stay in data/processed/variants/.
"""

from __future__ import annotations

import argparse
import glob

import pandas as pd
from src.config import REPO_ROOT, load_config
from src.data.features import (
    Calendar,
    build_season_features,
    chirps_daily,
    era5_daily,
    modis_clean,
)

IN = REPO_ROOT / "data" / "interim" / "ee_masked"
OUT = REPO_ROOT / "data" / "processed" / "masked"
KEY = {"modis": ["adm2_pcode", "date"], "chirps": ["adm2_pcode", "start"],
       "era5": ["adm2_pcode", "date"], "static": ["adm2_pcode"]}


def load(season: str, what: str) -> pd.DataFrame | None:
    pat = "static.csv" if what == "static" else f"{what}_*.csv"
    files = sorted(glob.glob(str(IN / season / "*" / pat)))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.drop_duplicates(subset=KEY[what], keep="last")


def load_inputs(season: str) -> dict:
    """Masked series for one season mask, cleaned with the pilot functions."""
    chirps, modis, era5 = load(season, "chirps"), load(season, "modis"), load(season, "era5")
    static = load(season, "static")
    if modis is None and chirps is None:
        raise SystemExit(f"no masked series for {season}; run src.data.ee_extract_masked")
    codes = sorted(set(modis.adm2_pcode) if modis is not None else set(chirps.adm2_pcode))
    return {
        "codes": codes,
        "rain": chirps_daily(chirps) if chirps is not None else None,
        "modis": modis_clean(modis, static) if modis is not None else None,
        "era5": era5_daily(era5) if era5 is not None else None,
        "first_year": int(chirps.start.str[:4].astype(int).min()) if chirps is not None else 2000,
    }


def build(lead: int, cfg: dict) -> pd.DataFrame:
    cal = Calendar.from_cfg(cfg)
    labels = pd.read_csv(REPO_ROOT / "data" / "interim" / "dcs_yield_tidy.csv")
    first = cfg["data"]["modelling_first_season"]
    parts = []
    for season in ("Maha", "Yala"):
        inp = load_inputs(season)
        pc = (labels[labels.adm2_pcode.isin(inp["codes"])]
              .drop_duplicates("district")[["district", "adm2_pcode"]])
        y1 = int(labels.season_year.max())
        grid = pd.DataFrame([(d, p, season, y) for d, p in pc.itertuples(index=False)
                             for y in range(inp["first_year"], y1 + 1)],
                            columns=["district", "adm2_pcode", "season", "season_year"])
        f = build_season_features(grid, labels, inp["rain"], inp["modis"], cal, lead, 0,
                                  era5=inp["era5"], heat_thr=cfg["features"]["heat_day_tmax_c"])
        parts.append(f)
    feats = pd.concat(parts, ignore_index=True)

    keep = ["district", "adm2_pcode", "season", "season_label", "season_year", "harvest_year",
            "season_order", "yield_kg_ha", "mahaweli_merged", "sanity_flag"]
    m = labels[keep].merge(feats, on=["district", "season", "season_year"], how="inner")
    m = m[m.season_year >= m.season.map(first)].reset_index(drop=True)
    forbidden = set(cfg["forbidden_features"])
    feat_cols = [c for c in feats.columns
                 if c not in ("district", "season", "season_year", "forecast_date",
                              "lead_weeks", "shift_weeks")]
    assert not forbidden & set(feat_cols), forbidden & set(feat_cols)
    m["mask"] = "static_2000_2008"
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", nargs="+", type=int, default=[12, 8, 4])
    a = ap.parse_args()
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    for lead in a.leads:
        m = build(lead, cfg)
        out = OUT / f"features_lead{lead}.parquet"
        m.to_parquet(out, index=False)
        print(f"wrote {out.relative_to(REPO_ROOT)}: {m.shape[0]} rows x {m.shape[1]} cols, "
              f"districts {sorted(m.district.unique())}")


if __name__ == "__main__":
    main()
