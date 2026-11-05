"""Step 2: seasonal active paddy mask (Earth Engine), Xiao et al. (2005) style, simplified.

Run on the local machine (Earth Engine auth already stored):

    # 1. one district, one season (test; writes to data/interim/ee/mask_test/)
    .venv/Scripts/python -m src.data.paddy_mask --districts LK72 --season Maha --seasons 2020-2020
    # 2. all 7 districts, 2020-2021
    .venv/Scripts/python -m src.data.paddy_mask --seasons 2020-2021
    # 3. everything (resumable)
    .venv/Scripts/python -m src.data.paddy_mask

For every season (Maha or Yala of one season_year) it builds one mask image and sums its
area inside each selected district. One request per season covers the 7 districts, and a
season whose CSV already exists is skipped, so a timeout only repeats that season.
Output: data/interim/ee/mask/mask_area_<season>_<season_year>.csv

The mask (evaluated on the MODIS 250 m grid, inside WorldCover cropland; the indices come
from 500 m MOD09A1):
1. Flooding: in the flood window, at least one clear observation with
   LSWI + margin >= EVI and EVI < flood_evi_max (water or wet soil at transplanting).
2. Green-up: in the grow window, max EVI >= peak_evi_min and
   (grow max EVI - flood-window min EVI) >= greenup_amplitude_min.
3. Harvest: (grow max EVI - harvest-window min EVI) >= harvest_drop_min.
Abandoned fields, wetlands and other evergreen vegetation stay green all season and fail
rules 2-3; permanent water fails rule 2.

Two variants are written because the Maha flood window is often cloudy:
- strict:  all three rules.
- relaxed: rule 1 is also accepted when the pixel has no clear observation in the flood
           window (rules 2 and 3 still apply).
Thresholds and windows live in configs/config.yaml (features.paddy_mask) and were fixed
before any comparison with DCS sown extent.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import pandas as pd

from src.config import REPO_ROOT, load_config
from src.data.ee_extract import (MODIS_ID, cropland_fraction, ee_init, get_with_retry,
                                 log)

OUT = REPO_ROOT / "data" / "interim" / "ee" / "mask"
OUT_TEST = REPO_ROOT / "data" / "interim" / "ee" / "mask_test"
SLOPE = REPO_ROOT / "data" / "interim" / "ee" / "district_slope.csv"
SCALE_M = 250          # MODIS MOD13Q1 grid, the same grid the pilot features use
TILE_SCALE = 8
GEOJSON = REPO_ROOT / "data" / "interim" / "districts_adm2_simplified.geojson"
MOD09A1_START = dt.date(2000, 2, 18)


# ---------------------------------------------------------------- season dates (pure Python)
def season_anchor(cfg: dict, season: str, season_year: int) -> dt.date:
    cal = cfg["features"]["season_calendar"][season]
    return dt.date(season_year, cal["anchor_month"], cal["anchor_day"])


def season_windows(cfg: dict, season: str, season_year: int) -> dict[str, tuple[dt.date, dt.date]]:
    """Flood, grow and harvest windows as [start, end) dates."""
    pm = cfg["features"]["paddy_mask"]
    a = season_anchor(cfg, season, season_year)
    out = {}
    for name in ("flood", "grow", "harvest"):
        d0, d1 = pm[f"{name}_window_days"]
        out[name] = (a + dt.timedelta(days=d0), a + dt.timedelta(days=d1))
    return out


# ---------------------------------------------------------------- Earth Engine pieces
def _prep(ee, img):
    """Scale reflectance, keep clear pixels, add EVI and LSWI."""
    qa = img.select("StateQA")
    clear = qa.bitwiseAnd(3).eq(0).And(qa.bitwiseAnd(4).eq(0))      # cloud state clear, no shadow
    r = img.select(["sur_refl_b01", "sur_refl_b02", "sur_refl_b03", "sur_refl_b06"]) \
        .multiply(0.0001).rename(["red", "nir", "blue", "swir1"])
    evi = r.expression("2.5*(n-r)/(n+6*r-7.5*b+1)",
                       {"n": r.select("nir"), "r": r.select("red"), "b": r.select("blue")})
    lswi = r.normalizedDifference(["nir", "swir1"])
    return ee.Image(evi.rename("evi").addBands(lswi.rename("lswi"))
                    .updateMask(clear).copyProperties(img, ["system:time_start"]))


def build_mask(ee, cfg: dict, season: str, season_year: int):
    """Return (strict, relaxed, flood_obs, cropland, grid) images.

    Everything is evaluated on the MODIS 250 m grid (`grid`, the MOD13Q1 projection).
    Cropland is the pilot's cropland_fraction(): WorldCover class 40 averaged from 10 m
    to that 250 m grid with reduceResolution (25 x 25 = 625 input pixels per output pixel,
    under the 1,024 limit). No other image is reprojected; the 500 m MOD09A1 indices are
    read at 250 m by the reducers (nearest-neighbour), so each 500 m value covers four
    250 m cells.
    """
    pm = cfg["features"]["paddy_mask"]
    coll = ee.ImageCollection(pm["collection"])
    grid = ee.ImageCollection(MODIS_ID).first().select("NDVI").projection()
    w = season_windows(cfg, season, season_year)

    def win(name):
        a, b = w[name]
        return coll.filterDate(a.isoformat(), b.isoformat()).map(lambda i: _prep(ee, i))

    flood_c, grow_c, harv_c = win("flood"), win("grow"), win("harvest")
    margin, emax = pm["flood_lswi_margin"], pm["flood_evi_max"]

    flooded_any = flood_c.map(
        lambda i: i.select("lswi").add(margin).gte(i.select("evi"))
        .And(i.select("evi").lt(emax)).rename("flood")).max().unmask(0)
    flood_obs = flood_c.select("evi").count().unmask(0).rename("flood_obs")
    flood_min = flood_c.select("evi").min()
    grow_max = grow_c.select("evi").max()
    harv_min = harv_c.select("evi").min()

    green = grow_max.gte(pm["peak_evi_min"]).And(
        grow_max.subtract(flood_min).gte(pm["greenup_amplitude_min"]))
    # no clear flood-window data -> no flood_min: judge green-up on the peak only
    green_relaxed = grow_max.gte(pm["peak_evi_min"]).And(
        grow_max.subtract(flood_min).gte(pm["greenup_amplitude_min"]).unmask(1))
    harvest = grow_max.subtract(harv_min).gte(pm["harvest_drop_min"])

    crop = cropland_fraction(ee, grid).gte(pm["cropland_min_fraction"]).unmask(0)
    strict = flooded_any.eq(1).And(green).And(harvest).unmask(0).And(crop)
    relaxed = (flooded_any.eq(1).Or(flood_obs.eq(0))).And(green_relaxed).And(harvest) \
        .unmask(0).And(crop)
    return (strict.rename("strict"), relaxed.rename("relaxed"),
            flood_obs, crop.rename("cropland"), grid)


def season_area(ee, cfg: dict, fc, season: str, season_year: int) -> list[dict]:
    """Mask and cropland area (ha) per district for one season: one request."""
    strict, relaxed, flood_obs, crop, grid = build_mask(ee, cfg, season, season_year)
    ha = ee.Image.pixelArea().divide(1e4)
    clear = flood_obs.gt(0)
    img = (ha.multiply(strict).rename("strict_ha")
           .addBands(ha.multiply(relaxed).rename("relaxed_ha"))
           .addBands(ha.multiply(crop).rename("cropland_ha"))
           .addBands(ha.multiply(crop).multiply(clear).rename("crop_clear_ha")))
    sums = img.reduceRegions(collection=fc, reducer=ee.Reducer.sum(),
                             scale=SCALE_M, crs=grid.crs(), tileScale=TILE_SCALE)
    a = get_with_retry(sums.map(lambda f: ee.Feature(None, f.toDictionary())),
                       f"mask {season} {season_year}")
    rows = []
    for f in a["features"]:
        p = f["properties"]
        crop_ha = p.get("cropland_ha")
        clear_ha = p.get("crop_clear_ha")
        rows.append({"adm2_pcode": p["adm2_pcode"], "season": season,
                     "season_year": season_year,
                     "mask_strict_ha": p.get("strict_ha"),
                     "mask_relaxed_ha": p.get("relaxed_ha"),
                     "cropland250_ha": crop_ha,
                     "flood_clear_share": (clear_ha / crop_ha) if crop_ha else None})
    return rows


# ---------------------------------------------------------------- driver
def selected_codes(cfg: dict) -> list[str]:
    sel = pd.read_csv(REPO_ROOT / cfg["data"]["selected_districts_file"])
    return sel["adm2_pcode"].tolist()


def seasons_to_run(cfg: dict, codes: list[str], years: tuple[int, int] | None):
    """Seasons that have DCS sown extent for the selected districts and full MODIS cover."""
    d = pd.read_csv(REPO_ROOT / "data" / "interim" / "dcs_yield_tidy.csv")
    d = d[d["adm2_pcode"].isin(codes) & d["sown_ha"].notna()]
    out = []
    today = dt.date.today()
    for (season, sy) in sorted(set(zip(d["season"], d["season_year"])), key=lambda t: (t[1], t[0])):
        w = season_windows(cfg, season, int(sy))
        if w["flood"][0] < MOD09A1_START or w["harvest"][1] > today:
            continue
        if years and not (years[0] <= sy <= years[1]):
            continue
        out.append((season, int(sy)))
    return out


def check_against_cropland(rows: list[dict]) -> None:
    """Test-run check: mask area must not exceed cropland area."""
    wc = pd.read_csv(SLOPE).set_index("adm2_pcode")["worldcover_cropland_ha"]
    for r in rows:
        code = r["adm2_pcode"]
        m, rel, c250 = r["mask_strict_ha"], r["mask_relaxed_ha"], r["cropland250_ha"]
        c10 = wc.get(code)
        ok = (m <= c250 + 1e-6) and (rel <= c250 + 1e-6)
        log(f"check {code} {r['season']} {r['season_year']}: strict {m:,.0f} ha, "
            f"relaxed {rel:,.0f} ha, cropland 250 m grid {c250:,.0f} ha, "
            f"WorldCover 10 m cropland {c10:,.0f} ha, flood_clear_share "
            f"{r['flood_clear_share']:.2f} -> {'OK' if ok else 'FAIL: mask > cropland'}")
        if not ok:
            raise RuntimeError(f"mask larger than cropland for {code}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", help="season_year range, e.g. 2020-2021")
    ap.add_argument("--season", choices=["Maha", "Yala"], help="only this season")
    ap.add_argument("--districts", help="comma-separated adm2 codes (test mode), e.g. LK72")
    args = ap.parse_args()
    years = tuple(int(x) for x in args.seasons.split("-")) if args.seasons else None

    cfg = load_config()
    codes = selected_codes(cfg)
    test = bool(args.districts)
    if test:
        want = [c.strip() for c in args.districts.split(",")]
        bad = set(want) - set(codes)
        if bad:
            raise RuntimeError(f"not in selected districts: {sorted(bad)}")
        codes = want
    out_dir = OUT_TEST if test else OUT

    ee = ee_init(cfg)
    gj = json.loads(GEOJSON.read_text(encoding="utf-8"))
    gj["features"] = [f for f in gj["features"] if f["properties"]["adm2_pcode"] in codes]
    if len(gj["features"]) != len(codes):
        raise RuntimeError("some selected districts are missing from the geojson")
    for f in gj["features"]:
        f["properties"] = {"adm2_pcode": f["properties"]["adm2_pcode"]}
    fc = ee.FeatureCollection(gj)

    out_dir.mkdir(parents=True, exist_ok=True)
    todo = seasons_to_run(cfg, codes, years)
    if args.season:
        todo = [t for t in todo if t[0] == args.season]
    log(f"mask: {len(todo)} seasons x {len(codes)} districts -> {out_dir.name}")
    for season, sy in todo:
        out = out_dir / f"mask_area_{season}_{sy}.csv"
        if out.exists() and not test:
            continue
        rows = season_area(ee, cfg, fc, season, sy)
        check_against_cropland(rows)
        pd.DataFrame(rows).to_csv(out, index=False)
        tot = sum(r["mask_strict_ha"] or 0 for r in rows)
        log(f"mask {season} {sy}: strict total {tot:,.0f} ha")
    log("mask: done")


if __name__ == "__main__":
    main()
