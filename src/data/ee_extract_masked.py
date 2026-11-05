"""Step 3b: MODIS, CHIRPS and ERA5-Land district series INSIDE the static paddy mask.

Run after `python -m src.data.static_mask export`:

    # first 3 districts, then the rest (files are per season-mask, dataset and year;
    # existing files are skipped, the current year is always refreshed)
    .venv/Scripts/python -m src.data.ee_extract_masked --districts LK72,LK71,LK52
    .venv/Scripts/python -m src.data.ee_extract_masked
    .venv/Scripts/python -m src.data.ee_extract_masked --what modis --years 2020-2021

Output: data/interim/ee_masked/<Maha|Yala>/{static,modis_YYYY,chirps_YYYY,era5_YYYY}.csv
with the same columns as the pilot files in data/interim/ee/, so src/data/features.py is
reused unchanged. A Maha row's features come from the Maha mask, a Yala row's from the Yala
mask.

How the mask enters each dataset
- MODIS MOD13Q1 (250 m): pixels outside the 0/1 mask are dropped (exact).
- CHIRPS (5.5 km) and ERA5-Land (11 km): the mask asset has mean pyramids, so at the
  coarse scale it is the paddy-area fraction of each coarse pixel and the district mean is
  paddy-area weighted. The pilot used a plain district mean for these.
Districts are processed in groups (one request per dataset x year x quarter for ERA5), with
retries for transient errors and fail-fast for deterministic ones.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time

import pandas as pd

from src.config import REPO_ROOT, load_config
from src.data.ee_extract import (
    CHIRPS_ID,
    ERA5_BANDS,
    ERA5_ID,
    MODIS_ID,
    DeterministicEEError,
    ee_init,
    get_with_retry,
    log,
)
from src.data.paddy_mask import GEOJSON, selected_codes
from src.data.static_mask import static_mask_image

OUT = REPO_ROOT / "data" / "interim" / "ee_masked"


def _fc(ee, codes):
    gj = json.loads(GEOJSON.read_text(encoding="utf-8"))
    gj["features"] = [f for f in gj["features"] if f["properties"]["adm2_pcode"] in codes]
    if len(gj["features"]) != len(codes):
        raise RuntimeError("some districts missing from the geojson")
    for f in gj["features"]:
        f["properties"] = {"adm2_pcode": f["properties"]["adm2_pcode"]}
    return ee.FeatureCollection(gj)


def _rows(info) -> pd.DataFrame:
    return pd.DataFrame([f["properties"] for f in info["features"]])


def extract_static(ee, fc, mask, cfg) -> pd.DataFrame:
    """Mask pixel count and area per district (denominator for MODIS valid fraction)."""
    img = mask.rename("crop").addBands(ee.Image.pixelArea().divide(1e6).multiply(mask).rename("km2"))
    res = img.reduceRegions(collection=fc, reducer=ee.Reducer.sum(), scale=250,
                            tileScale=cfg["earth_engine"]["tile_scale"])
    return _rows(get_with_retry(res.map(lambda f: ee.Feature(None, f.toDictionary())), "static"))


def extract_modis_year(ee, fc, mask, cfg, year: int) -> pd.DataFrame:
    col = ee.ImageCollection(MODIS_ID).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
    qa_max = cfg["earth_engine"]["modis_good_qa_max"]
    reducer = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)
    keep = mask.eq(1)

    def per_image(img):
        vi = (img.select(["NDVI", "EVI"]).multiply(0.0001)
              .updateMask(img.select("SummaryQA").lte(qa_max)).updateMask(keep))
        date = img.date().format("YYYY-MM-dd")
        stats = vi.reduceRegions(collection=fc, reducer=reducer, scale=250,
                                 tileScale=cfg["earth_engine"]["tile_scale"])
        return stats.map(lambda f: ee.Feature(None, f.toDictionary()).set("date", date))

    return _rows(get_with_retry(col.map(per_image).flatten(), f"modis {year}"))


def extract_chirps_year(ee, fc, mask, cfg, year: int) -> pd.DataFrame:
    col = ee.ImageCollection(CHIRPS_ID).filterDate(f"{year}-01-01", f"{year + 1}-01-01")

    def per_image(img):
        start = img.date().format("YYYY-MM-dd")
        end = ee.Date(img.get("system:time_end")).format("YYYY-MM-dd")
        stats = img.select("precipitation").updateMask(mask).reduceRegions(
            collection=fc, reducer=ee.Reducer.mean(),
            scale=cfg["earth_engine"]["scale_m"]["chirps"],
            tileScale=cfg["earth_engine"]["tile_scale"])
        return stats.map(lambda f: ee.Feature(None, f.toDictionary())
                         .set({"start": start, "end": end}))

    return _rows(get_with_retry(col.map(per_image).flatten(), f"chirps {year}"))


def extract_era5_year(ee, fc, mask, cfg, year: int) -> pd.DataFrame:
    parts = []
    for q0, q1 in (("01-01", "04-01"), ("04-01", "07-01"), ("07-01", "10-01"), ("10-01", None)):
        end = f"{year}-{q1}" if q1 else f"{year + 1}-01-01"
        col = ee.ImageCollection(ERA5_ID).filterDate(f"{year}-{q0}", end).select(ERA5_BANDS)

        def per_image(img):
            date = img.date().format("YYYY-MM-dd")
            stats = img.updateMask(mask).reduceRegions(
                collection=fc, reducer=ee.Reducer.mean(),
                scale=cfg["earth_engine"]["scale_m"]["era5_land"],
                tileScale=cfg["earth_engine"]["tile_scale"])
            return stats.map(lambda f: ee.Feature(None, f.toDictionary()).set("date", date))

        parts.append(_rows(get_with_retry(col.map(per_image).flatten(), f"era5 {year} {q0}")))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all", choices=["all", "modis", "chirps", "era5"])
    ap.add_argument("--years", help="e.g. 2000-2002")
    ap.add_argument("--districts", help="comma-separated adm2 codes, default: all selected")
    ap.add_argument("--seasons", default="Maha,Yala", help="which season masks")
    a = ap.parse_args()

    cfg = load_config()
    codes = selected_codes(cfg)
    if a.districts:
        want = [c.strip() for c in a.districts.split(",")]
        bad = set(want) - set(codes)
        if bad:
            raise RuntimeError(f"not in selected districts: {sorted(bad)}")
        codes = want
    tag = "_".join(sorted(codes)) if a.districts else "all"
    ee = ee_init(cfg)
    fc = _fc(ee, codes)
    this_year = dt.date.today().year
    eec = cfg["earth_engine"]
    ranges = {"modis": (eec["modis_first_year"], this_year),
              "chirps": (eec["chirps_first_year"], this_year),
              "era5": (eec["era5_first_year"], this_year)}
    if a.years:
        lo, _, hi = a.years.partition("-")
        ranges = {k: (int(lo), int(hi or lo)) for k in ranges}
    fns = {"modis": extract_modis_year, "chirps": extract_chirps_year, "era5": extract_era5_year}
    whats = list(fns) if a.what == "all" else [a.what]

    failed = []
    for season in a.seasons.split(","):
        mask = static_mask_image(ee, cfg, season)
        out_dir = OUT / season / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        sp = out_dir / "static.csv"
        if not sp.exists():
            extract_static(ee, fc, mask, cfg).to_csv(sp, index=False)
            log(f"masked {season} {tag}: static written")
        for what in whats:
            y0, y1 = ranges[what]
            for y in range(y0, y1 + 1):
                out = out_dir / f"{what}_{y}.csv"
                if out.exists() and y < this_year:
                    continue
                t0 = time.time()
                try:
                    df = fns[what](ee, fc, mask, cfg, y)
                except DeterministicEEError:
                    raise
                except Exception as e:
                    log(f"masked {season} {what} {y}: FAILED ({str(e)[:200]})")
                    failed.append(f"{season}/{what}_{y}")
                    continue
                df.to_csv(out, index=False)
                log(f"masked {season} {what} {y} [{tag}]: {len(df)} rows in {time.time() - t0:.0f}s")
    log(f"masked extract done. failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
