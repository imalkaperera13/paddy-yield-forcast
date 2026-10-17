"""Earth Engine extraction: district-level MODIS NDVI/EVI and CHIRPS rainfall time series.

Run on the local machine (needs `earthengine authenticate` once):

    python -m src.data.ee_extract                 # everything (resumable)
    python -m src.data.ee_extract --what modis --years 2000-2002
    python -m src.data.ee_extract --what chirps

Writes small CSVs to data/interim/ee/ (one per dataset-year) plus a log. A year whose file
already exists is skipped, except the current year, which is always refreshed. Nothing here
aggregates to seasons; that happens locally in src/data/features.py so that windows, shifts
and lead times can change without re-extracting.

Design (docs/decisions.md):
- Districts: HDX COD-AB adm2 polygons, simplified (100 m) and sent inline as GeoJSON.
- Cropland mask: ESA WorldCover v200 (2021) class 40, aggregated to the MODIS 250 m grid as
  a cropland fraction; MODIS pixels with fraction >= cropland_min_fraction are used. This is a
  proxy for paddy (WorldCover has no paddy class).
- MODIS MOD13Q1 v061: NDVI/EVI x 0.0001, SummaryQA <= 1 (good/marginal) only.
- CHIRPS pentad (mm per pentad), plain district mean (5.5 km pixels; no cropland weighting).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import zipfile
from pathlib import Path

import pandas as pd

from src.config import REPO_ROOT, load_config

OUT = REPO_ROOT / "data" / "interim" / "ee"
LOG = OUT / "extract_log.txt"
MODIS_ID = "MODIS/061/MOD13Q1"
CHIRPS_ID = "UCSB-CHG/CHIRPS/PENTAD"
ERA5_ID = "ECMWF/ERA5_LAND/DAILY_AGGR"
# Only total evaporation is used: the component evaporation bands are mislabelled in the
# source (known ECMWF issue noted in the EE catalog).
ERA5_BANDS = ["temperature_2m", "temperature_2m_max", "temperature_2m_min",
              "dewpoint_temperature_2m", "surface_solar_radiation_downwards_sum",
              "total_evaporation_sum"]
WORLDCOVER_ID = "ESA/WorldCover/v200"


def log(msg: str) -> None:
    line = f"{dt.datetime.now().isoformat(timespec='seconds')}  {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- districts (local, no EE)
def find_adm2_shp(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as z:
        shps = [n for n in z.namelist() if n.lower().endswith(".shp")]
    cand = [n for n in shps if "adm2" in n.lower() or "admin2" in n.lower()]
    if len(cand) != 1:
        raise RuntimeError(f"cannot pick the adm2 layer from {shps}")
    return cand[0]


def load_districts(cfg: dict):
    import geopandas as gpd

    raw = REPO_ROOT / cfg["paths"]["raw"] / "boundaries" / "lka_admin_boundaries.shp.zip"
    inner = find_adm2_shp(raw)
    log(f"boundaries: {raw.name} -> layer {inner}")
    # copy only the adm2 layer's sidecar files out of the zip (raw zip stays untouched)
    dest = REPO_ROOT / "data" / "interim" / "boundaries_adm2"
    dest.mkdir(parents=True, exist_ok=True)
    stem = inner[:-4]
    with zipfile.ZipFile(raw) as z:
        for n in z.namelist():
            if n.startswith(stem + "."):
                (dest / Path(n).name).write_bytes(z.read(n))
    gdf = gpd.read_file(dest / Path(inner).name)
    cols = {c.lower(): c for c in gdf.columns}
    gdf = gdf.rename(columns={cols["adm2_pcode"]: "adm2_pcode", cols["adm2_name"]: "adm2_name"})
    gdf = gdf[["adm2_pcode", "adm2_name", "geometry"]].to_crs(32644)        # UTM 44N, metres
    tol = cfg["earth_engine"]["district_simplify_m"]
    gdf["geometry"] = gdf.geometry.simplify(tol, preserve_topology=True)
    gdf = gdf.to_crs(4326)
    if len(gdf) != 25:
        raise RuntimeError(f"expected 25 districts, got {len(gdf)}")
    path = REPO_ROOT / "data" / "interim" / "districts_adm2_simplified.geojson"
    gdf.to_file(path, driver="GeoJSON")
    n_vert = int(gdf.geometry.apply(lambda g: len(g.exterior.coords) if g.geom_type == "Polygon"
                                    else sum(len(p.exterior.coords) for p in g.geoms)).sum())
    log(f"districts: {len(gdf)} polygons, {n_vert} vertices after {tol} m simplification")
    return gdf


# ---------------------------------------------------------------- Earth Engine helpers
def ee_init(cfg: dict):
    import ee

    ee.Initialize(project=cfg["earth_engine"]["project"])
    return ee


def to_fc(ee, gdf):
    return ee.FeatureCollection(json.loads(gdf[["adm2_pcode", "geometry"]].to_json()))


def get_with_retry(obj, what: str, tries: int = 4):
    for k in range(tries):
        try:
            return obj.getInfo()
        except Exception as e:  # EE raises EEException / HttpError; retry all
            wait = 15 * (2 ** k)
            log(f"  {what}: attempt {k + 1} failed ({str(e)[:160]}); retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"{what}: giving up after {tries} attempts")


def cropland_fraction(ee, modis_proj):
    wc = ee.ImageCollection(WORLDCOVER_ID).first().select("Map")
    return (wc.eq(40).reduceResolution(ee.Reducer.mean(), maxPixels=1024)
            .reproject(crs=modis_proj).rename("crop_frac"))


def extract_static(ee, fc, cfg) -> None:
    """Per-district cropland pixel count at 250 m (denominator for valid-pixel fraction)."""
    out = OUT / "static_cropland.csv"
    if out.exists():
        log("static: exists, skipped")
        return
    proj = ee.ImageCollection(MODIS_ID).first().projection()
    thr = cfg["features"]["cropland_mask"].get("min_fraction", 0.5)
    crop = cropland_fraction(ee, proj).gte(thr).selfMask().rename("crop")
    area_km2 = ee.Image.pixelArea().divide(1e6).rename("km2").updateMask(crop)
    img = crop.addBands(area_km2)       # sum of 'crop' (=1) = pixel count; sum of km2 = area
    stats = img.reduceRegions(collection=fc, reducer=ee.Reducer.sum(), scale=250,
                              tileScale=cfg["earth_engine"]["tile_scale"])
    rows = get_with_retry(stats.map(lambda f: ee.Feature(None, f.toDictionary())), "static")
    df = pd.DataFrame([r["properties"] for r in rows["features"]])
    df.to_csv(out, index=False)
    log(f"static: {len(df)} districts, columns {list(df.columns)}")


def extract_modis_year(ee, fc, cfg, year: int) -> pd.DataFrame:
    col = ee.ImageCollection(MODIS_ID).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
    proj = ee.ImageCollection(MODIS_ID).first().projection()
    thr = cfg["features"]["cropland_mask"].get("min_fraction", 0.5)
    crop = cropland_fraction(ee, proj).gte(thr)
    qa_max = cfg["earth_engine"]["modis_good_qa_max"]
    reducer = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)

    def per_image(img):
        vi = (img.select(["NDVI", "EVI"]).multiply(0.0001)
              .updateMask(img.select("SummaryQA").lte(qa_max)).updateMask(crop))
        date = img.date().format("YYYY-MM-dd")
        stats = vi.reduceRegions(collection=fc, reducer=reducer, scale=250,
                                 tileScale=cfg["earth_engine"]["tile_scale"])
        return stats.map(lambda f: ee.Feature(None, f.toDictionary()).set("date", date))

    rows = get_with_retry(col.map(per_image).flatten(), f"modis {year}")
    return pd.DataFrame([r["properties"] for r in rows["features"]])


def extract_chirps_year(ee, fc, cfg, year: int) -> pd.DataFrame:
    col = ee.ImageCollection(CHIRPS_ID).filterDate(f"{year}-01-01", f"{year + 1}-01-01")

    def per_image(img):
        start = img.date().format("YYYY-MM-dd")
        end = ee.Date(img.get("system:time_end")).format("YYYY-MM-dd")
        stats = img.select("precipitation").reduceRegions(
            collection=fc, reducer=ee.Reducer.mean(),
            scale=cfg["earth_engine"]["scale_m"]["chirps"],
            tileScale=cfg["earth_engine"]["tile_scale"])
        return stats.map(lambda f: ee.Feature(None, f.toDictionary())
                         .set({"start": start, "end": end}))

    rows = get_with_retry(col.map(per_image).flatten(), f"chirps {year}")
    return pd.DataFrame([r["properties"] for r in rows["features"]])


def extract_era5_year(ee, fc, cfg, year: int) -> pd.DataFrame:
    """Daily district means. Pulled per quarter (25 districts x ~92 days < 5000 rows/call)."""
    parts = []
    for q0, q1 in (("01-01", "04-01"), ("04-01", "07-01"), ("07-01", "10-01"), ("10-01", None)):
        end = f"{year}-{q1}" if q1 else f"{year + 1}-01-01"
        col = ee.ImageCollection(ERA5_ID).filterDate(f"{year}-{q0}", end).select(ERA5_BANDS)

        def per_image(img):
            date = img.date().format("YYYY-MM-dd")
            stats = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(),
                                      scale=cfg["earth_engine"]["scale_m"]["era5_land"],
                                      tileScale=cfg["earth_engine"]["tile_scale"])
            return stats.map(lambda f: ee.Feature(None, f.toDictionary()).set("date", date))

        rows = get_with_retry(col.map(per_image).flatten(), f"era5 {year} {q0}")
        parts.append(pd.DataFrame([r["properties"] for r in rows["features"]]))
    return pd.concat(parts, ignore_index=True)


def run(what: str, years: list[int] | None) -> None:
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"=== start: what={what} years={years or 'config default'}")
    gdf = load_districts(cfg)
    ee = ee_init(cfg)
    fc = to_fc(ee, gdf)
    this_year = dt.date.today().year
    meta = {"extracted_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "datasets": {"modis": MODIS_ID, "chirps": CHIRPS_ID, "era5_land": ERA5_ID,
                         "worldcover": WORLDCOVER_ID},
            "ee_project": cfg["earth_engine"]["project"], "ee_api": ee.__version__}
    (OUT / "extract_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if what in ("all", "static"):
        extract_static(ee, fc, cfg)
    jobs = []
    if what in ("all", "modis"):
        ys = years or range(cfg["earth_engine"]["modis_first_year"], this_year + 1)
        jobs += [("modis", y, extract_modis_year) for y in ys]
    if what in ("all", "chirps"):
        ys = years or range(cfg["earth_engine"]["chirps_first_year"], this_year + 1)
        jobs += [("chirps", y, extract_chirps_year) for y in ys]
    if what in ("all", "era5"):
        ys = years or range(cfg["earth_engine"]["era5_first_year"], this_year + 1)
        jobs += [("era5", y, extract_era5_year) for y in ys]
    failed = []
    for name, y, fn in jobs:
        out = OUT / f"{name}_{y}.csv"
        if out.exists() and y < this_year:
            continue
        t0 = time.time()
        try:
            df = fn(ee, fc, cfg, y)
        except Exception as e:
            log(f"{name} {y}: FAILED ({str(e)[:200]})")
            failed.append(f"{name}_{y}")
            continue
        df.to_csv(out, index=False)
        log(f"{name} {y}: {len(df)} rows in {time.time() - t0:.0f}s")
    log(f"=== done. failed: {failed or 'none'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all", choices=["all", "static", "modis", "chirps", "era5"])
    ap.add_argument("--years", help="e.g. 2000-2002 or 2005")
    a = ap.parse_args()
    yrs = None
    if a.years:
        lo, _, hi = a.years.partition("-")
        yrs = list(range(int(lo), int(hi or lo) + 1))
    run(a.what, yrs)
