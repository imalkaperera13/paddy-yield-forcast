# Methods (draft for the report)

> **Status: complete for Stage 1 (2026-10-18).** Data versions and download dates come from `docs/data_dictionary.md`, and run versions come from `docs/decisions.md`.

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
All models use the same 120 features plus a Maha/Yala indicator and are evaluated through the same splits (`src/models/registry.py`):
- **Baseline:** for each district, the mean of the last 5 published same-season yields (`y_last5_same`). No training.
- **Ridge regression:** median imputation and standardisation; α ∈ {0.1, 1, 10, 100, 1000}.
- **Random forest:** 200 trees, median imputation, max_features = 0.5; min_samples_leaf ∈ {5, 15}.
- **LightGBM:** learning rate 0.03, subsample 0.8, colsample 0.7, λ = 1, min_child_samples = 20; n_estimators ∈ {200, 500}, num_leaves ∈ {7, 15}.
- **XGBoost:** learning rate 0.03, subsample 0.8, colsample 0.7, λ = 1, min_child_weight = 5, histogram method; n_estimators ∈ {200, 500}, max_depth ∈ {3, 5}.
- **Hybrid (baseline + residual):** LightGBM fitted to *y − baseline*; the prediction is baseline + model output. n_estimators ∈ {100, 300}, num_leaves = 7.

The grids are deliberately small given about 1,000 training rows. Seeds: 42, 7, 123, 2024 and 31337; results are reported as mean ± standard deviation over seeds.

## 4. Validation protocol
- **Forward-chaining by season** (main results). Each fold predicts one season (all districts). Training uses only seasons whose yields would have been published by that season's forecast date (reference harvest + an assumed 6-month lag). The first test season is 2009/10 Maha, the first with ≥ 10 harvest years of training data; this gives 33 folds and 808 test district-seasons.
- **Hyper-parameter tuning** happens inside each training window only: an expanding window over its last 3 harvest years, choosing the setting with the lowest mean absolute error. The tuned model is then refitted on the whole window.
- **Leave-district-out**: the same folds, with the test district removed from training (spatial generalisation). Fixed parameters are used here to limit compute.
- **Calibration split** (conformal intervals): the most recent 3 harvest years inside each training window.
- All models are compared on the same district-seasons, namely those where every model and the baseline produce a prediction.

## 5. Metrics
Definitions, with $y_i$ the observed and $\hat y_i$ the predicted yield (kg/ha) over $n$ district-seasons and $\bar y$ the mean observed yield:

- RMSE $= \sqrt{\tfrac{1}{n}\sum_i (y_i-\hat y_i)^2}$; also reported as % of $\bar y$
- MAE $= \tfrac{1}{n}\sum_i |y_i-\hat y_i|$; also reported as % of $\bar y$
- MAPE $= \tfrac{100}{n}\sum_i |y_i-\hat y_i| / y_i$
- $R^2 = 1 - \sum_i (y_i-\hat y_i)^2 / \sum_i (y_i-\bar y)^2$
- Interval coverage $= \tfrac{1}{n}\sum_i \mathbb{1}[\,lo_i \le y_i \le hi_i\,]$; mean width $= \tfrac{1}{n}\sum_i (hi_i - lo_i)$

## 6. Statistical testing
For each model, the absolute errors per district-season (averaged over seeds) are compared with the baseline's absolute errors on the same district-seasons using a two-sided paired Wilcoxon signed-rank test (pairs with zero difference are dropped). The effect size is the matched-pairs rank-biserial correlation, *r* = (W⁺ − W⁻)/(W⁺ + W⁻); a positive *r* means the model's errors are smaller. p-values are Holm-adjusted across the five model-vs-baseline comparisons.

## 7. Prediction intervals (split conformal)
For each forward-chaining fold, the training window is split into a proper training set (all but the last 3 harvest years) and a calibration set (the last 3 harvest years). The model is tuned and fitted on the proper set only. The absolute residuals $r_i = |y_i - \hat y_i|$ on the $n$ calibration rows give
$$q = r_{(k)},\quad k = \lceil (n+1)(1-\alpha) \rceil,$$
the $k$-th smallest residual ($q=\infty$ if $k>n$). The interval for a test row is $\hat y \pm q$, with α = 0.1. A per-season (Mondrian) variant computes $q$ separately from Maha and Yala calibration rows. Reliability is assessed at α ∈ {0.5, 0.4, 0.3, 0.2, 0.1, 0.05}. The finite-sample guarantee assumes the calibration and test seasons are exchangeable. Seasonal yields are not (trend, policy shocks), so coverage is reported overall, per test season and inside vs outside pre-named shock seasons.

## 8. Earliness and window-shift sensitivity
Forecasts are made L ∈ {12, 10, 8, 6, 4, 2, 0} weeks before the reference harvest date. For each L the features are rebuilt so that only satellite and climate data ending before that date (minus the assumed latencies) enter. The forward-chaining folds are rebuilt so that training uses only labels published by that date. All leads are scored on the same district-seasons. Sensitivity to the nominal planting dates is tested by moving all growth windows ±2 weeks at lead 0.

## 9. Feature-group ablation and SHAP
The random forest is retrained on 9 feature sets: all groups; lagged yield only; lagged yield plus one other group; all minus one group; and no lagged yield. Each set is tuned within each training window and scored on the same rows. Each variant is compared with the baseline and with the full model by paired Wilcoxon tests, Holm-adjusted over all 17 comparisons. SHAP values (TreeExplainer) are computed for every test row from the model of its own fold, refitted with that fold's chosen parameters. Importance is the mean |SHAP| per feature, summed per group.

## 10. Shock seasons
Shock seasons were fixed before any error was inspected:
- Maha 2021/22, Yala 2022 and Maha 2022/23 (chemical-fertiliser ban and economic crisis);
- Maha 2010/11 (floods);
- the 3 other largest negative national area-weighted yield anomalies relative to the previous 10 same seasons (Maha 2023/24, Maha 2024/25 and Yala 2023; cause not verified).

Error and mean bias (predicted − observed) are reported per shock season and pooled, against all other test seasons, from the saved forward-chaining predictions.

## 11. Final models
For forecasting new seasons, the random forest is refitted on all 1,266 labelled district-seasons with two feature sets: all features, and lagged yield + MODIS. Hyper-parameters are tuned with the same inner time splits. A split-conformal $q$ is computed with the last 3 harvest years as calibration (model fitted on the older years). These fits are not used for any reported metric.

## 12. Computing environment
Training ran on Kaggle (CPU, internet off), in five kernels:
- `paddy-01-baselines-and-models` v2;
- `paddy-02-conformal` v1;
- `paddy-03-earliness` v1;
- `paddy-04-ablation-shap` v1;
- `paddy-05-final-model`.

Kernel 01 used code commit `3a1d358` and data `paddy-master-table` v1. Kernels 02–05 used commit `cd8d7da` and data v2. Both data versions contain the same master table (SHA-256 `2a8fdc80…`); v2 adds the lead and shift tables. Library versions come from each kernel's `run_info.json` (`results/runs_meta/`). Kaggle's image provided Python 3.13, scikit-learn 1.6.1, LightGBM 4.6.0, XGBoost 3.4.1 and SHAP 0.52.0, which differ from the pinned local `requirements.txt`. Shock analysis and report figures were computed locally from saved predictions (`python -m src.run_all`).
