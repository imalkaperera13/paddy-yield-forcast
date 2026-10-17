# Methods (draft for the report)

> **Status: skeleton.** Sections are filled in as each task is completed. Data versions and download dates come from `docs/data_dictionary.md`, and run versions come from `docs/decisions.md`.

## 1. Data
### 1.1 Yield labels: DCS Paddy Statistics
_pending (Task 1): source URL, download date, SHA-256, seasons, season-year convention, Mahaweli handling, dropped rows_
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
