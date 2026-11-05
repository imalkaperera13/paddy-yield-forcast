"""Step 1a: terrain slope over cropland for each district (Earth Engine).

Run on the local machine (Earth Engine auth already stored):

    .venv/Scripts/python -m scripts.district_slope

Output: data/interim/ee/district_slope.csv (one row per district, 25 rows).

Method
- Slope in degrees from SRTM 30 m (USGS/SRTMGL1_003) via ee.Terrain.slope.
- Cropland = ESA WorldCover v200 class 40, the same proxy the pilot used. WorldCover has no
  paddy class, so this only describes the land the pilot already treats as paddy.
- Statistics are computed at 90 m inside each district's cropland: mean slope, median slope,
  share of cropland steeper than 3 and 5 degrees, mean elevation and cropland area.
- One district per request, with the pilot's retry helper, so a timeout only repeats one district.
"""

from __future__ import annotations

import json

import pandas as pd

from src.config import REPO_ROOT, load_config
from src.data.ee_extract import WORLDCOVER_ID, ee_init, get_with_retry, log

SRTM_ID = "USGS/SRTMGL1_003"
GEOJSON = REPO_ROOT / "data" / "interim" / "districts_adm2_simplified.geojson"
OUT = REPO_ROOT / "data" / "interim" / "ee" / "district_slope.csv"
SCALE_M = 90


def main() -> None:
    cfg = load_config()
    ee = ee_init(cfg)
    tile = cfg["earth_engine"]["tile_scale"]

    gj = json.loads(GEOJSON.read_text(encoding="utf-8"))
    dem = ee.Image(SRTM_ID).select("elevation")
    slope = ee.Terrain.slope(dem)
    crop = ee.ImageCollection(WORLDCOVER_ID).first().select("Map").eq(40)

    img = (slope.rename("slope_deg")
           .addBands(slope.gt(3).rename("steep3"))
           .addBands(slope.gt(5).rename("steep5"))
           .addBands(dem.rename("elev_m"))
           .updateMask(crop))
    area = ee.Image.pixelArea().divide(1e4).rename("cropland_ha").updateMask(crop)

    reducer = ee.Reducer.mean()

    rows = []
    for feat in gj["features"]:
        code = feat["properties"]["adm2_pcode"]
        name = feat["properties"].get("adm2_name", "")
        geom = ee.Geometry(feat["geometry"])
        mean_d = img.reduceRegion(reducer, geom, SCALE_M, maxPixels=1e10, tileScale=tile)
        med_d = img.select("slope_deg").reduceRegion(ee.Reducer.median(), geom, SCALE_M,
                                                     maxPixels=1e10, tileScale=tile)
        area_d = area.reduceRegion(ee.Reducer.sum(), geom, SCALE_M, maxPixels=1e10,
                                   tileScale=tile)
        both = ee.Dictionary(mean_d).combine(
            ee.Dictionary({"slope_median_deg": med_d.get("slope_deg"),
                           "cropland_ha": area_d.get("cropland_ha")}))
        d = get_with_retry(both, f"slope {code}")
        rows.append({
            "adm2_pcode": code,
            "adm2_name": name,
            "crop_slope_mean_deg": d.get("slope_deg"),
            "crop_slope_median_deg": d.get("slope_median_deg"),
            "crop_share_gt3deg": d.get("steep3"),
            "crop_share_gt5deg": d.get("steep5"),
            "crop_elev_mean_m": d.get("elev_m"),
            "worldcover_cropland_ha": d.get("cropland_ha"),
        })
        log(f"slope {code} {name}: mean {d.get('slope_deg')}")

    df = pd.DataFrame(rows).sort_values("adm2_pcode")
    if len(df) != 25:
        raise RuntimeError(f"expected 25 districts, got {len(df)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    log(f"slope: wrote {OUT.relative_to(REPO_ROOT)} ({len(df)} rows)")
    print(df.sort_values("crop_slope_mean_deg").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
