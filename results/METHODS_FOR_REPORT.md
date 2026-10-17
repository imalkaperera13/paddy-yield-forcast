# Methods (draft for the report)

> **Status: skeleton.** Sections are filled in as each task is completed. Data versions and download dates come from `docs/data_dictionary.md`, and run versions come from `docs/decisions.md`.

## 1. Data
### 1.1 Yield labels: DCS Paddy Statistics
District-level average yield (kg per net hectare), production, and gross sown and harvested extent for the Maha (1978/79–2025/26) and Yala (1979–2025) seasons. Source: Department of Census and Statistics, Sri Lanka, Paddy Statistics series. The files were downloaded by hand on 2026-10-17; SHA-256 hashes are in `docs/raw_provenance.md`.

Maha seasons are labelled by planting year (Maha 2021/22 → 2021), and the harvest year is stored as well. Time ordering uses a harvest-based index, so each Maha season precedes the following Yala season.

The 25 administrative districts are used as published. The Mahaweli H and Udawalawe irrigation-area rows, reported separately until 2017/18 Maha and 2018 Yala, are excluded.

Three file defects were corrected in code:
- two mislabelled season headers;
- a district row-order mismatch in the Maha production series. This was resolved by agreement with yield × harvested area, so production-derived quantities are not independent of yield.

Published yields of 0 kg/ha accompanied by positive production were treated as not reported. Missing district-seasons (101 of 2,375) were dropped, not imputed.
### 1.2 District boundaries: HDX COD-AB Sri Lanka
Administrative level-2 (district) polygons, COD-AB v03 (valid from 2022-08-16), from the Humanitarian Data Exchange. The file was downloaded on 2026-10-17. The 25 districts were joined to the DCS names via P-codes (`configs/district_map.csv`) and simplified by 100 m before use in Google Earth Engine.

### 1.3 Satellite and climate data (Google Earth Engine, extracted 2026-10-17, earthengine-api 1.4.3)
- **MODIS MOD13Q1 v061** (`MODIS/061/MOD13Q1`): 16-day 250 m NDVI and EVI, 2000-02-18 to 2026-09-14, scaled by 0.0001. Only pixels with SummaryQA ≤ 1 (good or marginal) are kept. District values are means over cropland pixels, with the valid-pixel count recorded. 15,300 district-composites; 30 had no valid pixels.
- **CHIRPS v2.0 pentad** (`UCSB-CHG/CHIRPS/PENTAD`): ~5.5 km rainfall, 1981-01-01 to 2026-10-01, as district means (no cropland weighting). 82,350 district-pentads.
- **ERA5-Land daily aggregates** (`ECMWF/ERA5_LAND/DAILY_AGGR`): ~11 km, 1991-01-01 to 2026, as district means. The bands used are 2 m mean, maximum and minimum temperature; dewpoint; total incoming solar radiation; and total evaporation (sign flipped; the component evaporation bands are mislabelled in the source, so they are not used). Vapour-pressure deficit is computed from mean temperature and dewpoint using the Tetens formula.
- **Cropland mask:** ESA WorldCover v200 (2021) class 40 ("cropland"), aggregated to the MODIS grid. A 250 m pixel is treated as cropland if ≥ 50% of its 10 m cells are class 40. WorldCover has no paddy class, so this is a proxy. Its area is 0.85× the DCS Maha sown extent nationally, but only 0.10–0.47× in wet-zone districts.

## 2. Features
Each season is anchored at a nominal planting date: Maha 15 October, Yala 15 April. From the anchor come five consecutive 4-week growth blocks (b1–b5) and one pre-season block. The reference harvest date is anchor + 140 days. A forecast with lead time *L* weeks is made at reference harvest − *L*, and uses only data available at that date: CHIRPS pentads and MODIS composites count only if they end at least 5 days (an assumed latency) before it.

Per district-season, the features are:
- rainfall per block and season total;
- NDVI and EVI mean per block, season maximum, and the sum over composites;
- MODIS valid-pixel fraction per block;
- temperature (mean, max, min), vapour-pressure deficit, solar radiation and evaporation per block; season mean temperature, total radiation, and heat-stress days (daily max ≥ 35 °C). ERA5 days count only if they end at least 7 days (an assumed latency) before the forecast date;
- an anomaly for each of these vs the mean of the same district and season in strictly previous years (at least 3 years).

Lagged-yield features are the previous same-season yield, the mean of all and of the last 5 same-season yields, a 10-season linear trend value, and the most recent available yield and its age. They also include the previous same-season sown extent and harvested/sown ratio. Every past yield is used only if it would have been published, assuming a 6-month lag after its reference harvest date. Target-season production and extents are never used.

The modelling table covers 2000/01–2025/26 Maha (647 rows) and 2000–2025 Yala (619 rows; Jaffna has no Yala yield): 1,266 rows and 120 features (lagged yield 9, rainfall 14, MODIS 31, temperature/radiation 66). Earlier yields (1979–1999) are used only to build lags and baselines.

## 3. Models and hyper-parameters
_pending (Task 5)_

## 4. Validation protocol
_pending (Task 4): forward-chaining by season, leave-district-out, calibration split, inner time-aware tuning_

## 5. Metrics
Definitions, with $y_i$ the observed and $\hat y_i$ the predicted yield (kg/ha) over $n$ district-seasons and $\bar y$ the mean observed yield:

- RMSE $= \sqrt{\tfrac{1}{n}\sum_i (y_i-\hat y_i)^2}$; also reported as % of $\bar y$
- MAE $= \tfrac{1}{n}\sum_i |y_i-\hat y_i|$; also reported as % of $\bar y$
- MAPE $= \tfrac{100}{n}\sum_i |y_i-\hat y_i| / y_i$
- $R^2 = 1 - \sum_i (y_i-\hat y_i)^2 / \sum_i (y_i-\bar y)^2$
- Interval coverage $= \tfrac{1}{n}\sum_i \mathbb{1}[\,lo_i \le y_i \le hi_i\,]$; mean width $= \tfrac{1}{n}\sum_i (hi_i - lo_i)$

## 6. Statistical testing
_pending (Task 6): paired Wilcoxon signed-rank on per district-season absolute errors, effect size, multiple-comparison correction_

## 7. Computing environment
Training ran on Kaggle (CPU, internet off). Library versions are taken from each kernel's `run_info.json`.
