# Model card: district paddy yield forecaster (Sri Lanka)

## Model
- **Files:**
  - `rf_all.joblib`: random forest, 120 features + season flag. This is the main model.
  - `rf_lagged_modis.joblib`: random forest, lagged yield + MODIS (40 features + flag).
  - `final_models.json`: features, parameters and the conformal q.

  The `.joblib` files are not in git (`.gitignore`). Get them from the output of Kaggle kernel `imalkaperera/paddy-05-final-model` v1, or rebuild them with `python -m src.experiments.final_model`.
- **Trained by** Kaggle kernel `paddy-05-final-model` v1 (2026-10-18). Code commit `868c55f`, data `paddy-master-table` v2 (master SHA-256 `2a8fdc80…`).
- **Hyper-parameters:** 200 trees, `min_samples_leaf` 15, `max_features` 0.5, seed 42, median imputation. Tuned with expanding-window inner splits over the last 3 harvest years.
- **Library:** scikit-learn **1.6.1** (Kaggle image). Load it with the same version; the local pin in `requirements.txt` is 1.5.2, and pickles are not guaranteed to load across versions.
- **Baseline it must beat:** `y_last5_same`, the mean of the last 5 published same-season yields. This needs no model.

## Intended use
- **Purpose:** research and teaching. It shows a leakage-safe, calibrated forecasting pipeline for district-level Maha/Yala paddy yield (kg per net hectare) at or before harvest.
- **Not for** operational food-security decisions, insurance or policy without independent validation.

## Training data
- 1,266 district-seasons, 2000/01 Maha – 2025/26 Maha (harvest years 2000–2026), 25 districts.
- Labels: DCS Paddy Statistics (crop-cut estimates, which are noisy). Features: MODIS NDVI/EVI over a WorldCover cropland mask, CHIRPS rainfall and ERA5-Land; see `docs/data_card.md`.

## Evaluation
All figures come from honest forward-chaining validation (33 test seasons, 808 district-seasons), **not** from these final fits. See `results/RESULTS.md`.

| model | RMSE (kg/ha) | MAE | R² | source |
|---|---|---|---|---|
| baseline (last-5 mean) | 605.8 | 442.7 | 0.521 | kernel 01 v2 |
| random forest, all features (5 seeds) | 604.1 ± 2.9 | 435.0 ± 2.8 | 0.523 | kernel 01 v2 |
| random forest, lagged + MODIS (seed 42) | 596.4 | 430.4 | 0.535 | kernel 04 v1 |

- **No model is significantly better than the baseline** (paired Wilcoxon, Holm): the random forest has p = 0.45.
- **Earliness:** 12 weeks before harvest the random forest's RMSE is 614.9 (kernel 03).
- **90% split-conformal intervals:** coverage 0.861 overall; 0.903 outside and 0.566 inside the named shock seasons (kernel 02).

## Prediction intervals in this release
The conformal q in `final_models.json` uses the last 3 harvest years (2024–2026) as calibration: ±910.6 kg/ha (`rf_all`) and ±817.5 kg/ha (`rf_lagged_modis`) for α = 0.1. These calibration years follow the 2021–22 crisis, so the intervals are wide.

## Limitations and risks
- **Shocks:** the model does not see policy or economic shocks. In the 2021/22 fertiliser-ban season every model over-forecast by about 1,400–1,560 kg/ha, and interval coverage collapses in such seasons.
- **Weather signal:** rainfall and temperature/radiation features add no measurable skill at district scale. Skill comes mostly from the district's own yield history.
- **Cropland mask:** WorldCover has no paddy class, and wet-zone paddy is under-mapped (mask/area ratio 0.10–0.47).
- **Labels:** the label-publication lag (6 months) is an assumption. The Mahaweli H / Udawalawe merge (2017/18) is a structural break.
- **Coverage gap:** Jaffna has no Yala data.

## Usage
```python
from pathlib import Path
import pandas as pd
from src.experiments.final_model import predict
rows = pd.read_parquet("data/processed/master_table.parquet").tail(5)   # rows with features built
print(predict(Path("models/rf_all.joblib"), Path("models/final_models.json"), rows))
```
