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
_pending_
### 1.3 Satellite and climate features
_pending (Task 2): dataset IDs, versions, scale, temporal coverage, cropland mask (WorldCover class 40 as proxy)_

## 2. Features
_pending: growth-stage windows, aggregation, lagged yields (past seasons only)_

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
