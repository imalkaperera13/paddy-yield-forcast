"""Step 1b: choose the flat dry-zone districts PaddyCast starts with.

Run after scripts/district_slope.py:

    .venv/Scripts/python -m scripts.select_districts

Inputs
- data/interim/dcs_yield_tidy.csv           DCS sown extent (as published, 25 districts)
- data/interim/ee/district_slope.csv        SRTM slope over WorldCover cropland
Outputs
- results/tables/district_screening.csv     all 25 districts with every criterion
- data/processed/selected_districts.csv     districts that pass all rules

Rules (constants below, also written into the screening table)
1. Large paddy area: mean annual sown extent (Maha + Yala) over the last 10 harvest years
   >= MIN_ANNUAL_SOWN_HA.
2. Flat: mean slope over cropland <= MAX_MEAN_SLOPE_DEG and share of cropland steeper than
   5 degrees <= MAX_SHARE_GT5.
3. Dry zone. ASSUMPTION (general agro-climatic zoning, not checked against an official map
   in this repo): the list in DRY_ZONE. Kurunegala is mostly intermediate zone and is
   flagged, not excluded, so the choice stays with the user.
"""

from __future__ import annotations

import pandas as pd

from src.config import REPO_ROOT, load_config

DCS = REPO_ROOT / "data" / "interim" / "dcs_yield_tidy.csv"
SLOPE = REPO_ROOT / "data" / "interim" / "ee" / "district_slope.csv"
SCREEN = REPO_ROOT / "results" / "tables" / "district_screening.csv"
SELECTED = REPO_ROOT / "data" / "processed" / "selected_districts.csv"

YEARS = (2016, 2025)                 # harvest years used for the sown-extent average
MIN_ANNUAL_SOWN_HA = 50_000
MAX_MEAN_SLOPE_DEG = 3.0
MAX_SHARE_GT5 = 0.10

DRY_ZONE = {"ANURADHAPURA", "POLONNARUWA", "AMPARA", "BATTICALOA", "TRINCOMALEE",
            "HAMBANTOTA", "MONARAGALA", "MANNAR", "VAVUNIYA", "MULLAITIVU",
            "KILINOCHCHI", "JAFFNA", "PUTTALAM"}
INTERMEDIATE = {"KURUNEGALA"}


def sown_summary() -> pd.DataFrame:
    d = pd.read_csv(DCS)
    d = d[d["harvest_year"].between(*YEARS)]
    p = d.pivot_table(index=["district", "adm2_pcode"], columns="season",
                      values="sown_ha", aggfunc="mean")
    p.columns = [f"sown_{c.lower()}_ha_mean" for c in p.columns]
    p["annual_sown_ha_mean"] = p.sum(axis=1)
    p["n_seasons"] = d.groupby(["district", "adm2_pcode"]).size()
    return p.reset_index()


def main() -> None:
    cfg = load_config()
    mahaweli = set(cfg["data"].get("mahaweli_affected_districts", [])) \
        if "data" in cfg else set()
    if not mahaweli:
        mahaweli = {"ANURADHAPURA", "RATNAPURA", "MONARAGALA", "HAMBANTOTA"}

    s = sown_summary()
    sl = pd.read_csv(SLOPE)
    df = s.merge(sl, on="adm2_pcode", how="left", validate="one_to_one")
    if df["crop_slope_mean_deg"].isna().any():
        raise RuntimeError("slope missing for some districts; re-run scripts/district_slope.py")

    df["zone"] = df["district"].map(
        lambda x: "dry" if x in DRY_ZONE else ("intermediate" if x in INTERMEDIATE else "other"))
    df["mahaweli_affected"] = df["district"].isin(mahaweli)
    df["pass_area"] = df["annual_sown_ha_mean"] >= MIN_ANNUAL_SOWN_HA
    df["pass_flat"] = ((df["crop_slope_mean_deg"] <= MAX_MEAN_SLOPE_DEG)
                       & (df["crop_share_gt5deg"] <= MAX_SHARE_GT5))
    df["pass_zone"] = df["zone"].isin(["dry", "intermediate"])
    df["selected"] = df["pass_area"] & df["pass_flat"] & df["pass_zone"]
    df["rule"] = (f"sown>={MIN_ANNUAL_SOWN_HA} ha (mean {YEARS[0]}-{YEARS[1]}); "
                  f"mean slope<={MAX_MEAN_SLOPE_DEG} deg; share>5deg<={MAX_SHARE_GT5}; "
                  f"dry or intermediate zone")
    df = df.sort_values("annual_sown_ha_mean", ascending=False)

    SCREEN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SCREEN, index=False)
    keep = ["district", "adm2_pcode", "zone", "sown_maha_ha_mean", "sown_yala_ha_mean",
            "annual_sown_ha_mean", "crop_slope_mean_deg", "crop_slope_median_deg",
            "crop_share_gt5deg", "crop_elev_mean_m", "mahaweli_affected"]
    sel = df[df["selected"]][keep]
    SELECTED.parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(SELECTED, index=False)

    show = ["district", "zone", "annual_sown_ha_mean", "crop_slope_mean_deg",
            "crop_share_gt5deg", "pass_area", "pass_flat", "selected"]
    print(df[show].round(3).to_string(index=False))
    print(f"\nselected {len(sel)} districts -> {SELECTED.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
