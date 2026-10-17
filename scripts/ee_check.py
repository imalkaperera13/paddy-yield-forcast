"""Earth Engine connection test. Run after `earthengine authenticate`:

    .venv/Scripts/python scripts/ee_check.py

Reads one MODIS NDVI image and one day of CHIRPS rain over Sri Lanka.
Prints only dataset metadata and two numbers. No credentials are printed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import ee

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402

project = load_config()["earth_engine"]["project"]
ee.Initialize(project=project)
print(f"Earth Engine initialised with project: {project}")

sri_lanka = ee.Geometry.Rectangle([79.5, 5.8, 82.0, 9.9])

modis = (ee.ImageCollection("MODIS/061/MOD13Q1")
         .filterDate("2024-01-01", "2024-02-01").first())
ndvi = modis.select("NDVI").multiply(0.0001).reduceRegion(
    ee.Reducer.mean(), sri_lanka, scale=5000, maxPixels=1e9).get("NDVI")
print("MODIS MOD13Q1 date:", ee.Date(modis.get("system:time_start")).format("YYYY-MM-dd").getInfo())
print("  mean NDVI over Sri Lanka box (5 km test scale):", round(ndvi.getInfo(), 3))

rain = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate("2024-11-15", "2024-11-16").first())
mm = rain.reduceRegion(ee.Reducer.mean(), sri_lanka, scale=5566).get("precipitation")
print("CHIRPS 2024-11-15 mean rain (mm):", round(mm.getInfo(), 2))
print("OK: Earth Engine works.")
