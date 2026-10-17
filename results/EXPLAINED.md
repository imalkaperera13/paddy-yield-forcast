# Results explained

> **Status: kernel 01 results in (2026-10-17); later sections pending.** Every table and figure below is filled in only from code that has been run. Each one cites its Kaggle kernel version and git commit.

For each item, the explanation covers:
- what it shows,
- how to read it,
- what it does **not** prove,
- why it may have happened. These reasons are hypotheses unless they were tested.

## Model comparison
*Source: kernel `paddy-01-baselines-and-models` v2, commit `3a1d358`; table in `RESULTS.md` §1.*
- **What it shows:** the error of each model, forecasting at the reference harvest date, over 808 district-seasons in 33 test seasons (2009/10–2025/26).
- **How to read it:** lower RMSE/MAE is better. Errors average about 15% of mean yield (RMSE) and about 11% (MAE). The random forest (604 kg/ha RMSE) and the last-5-seasons baseline (606) are level. Ridge (620), XGBoost (633), LightGBM (634) and the hybrid (661) are all worse than the baseline.
- **What it does not prove:** that satellite/climate data carry no signal. It shows only that, with these features, about 25 seasons per district and these models, they do not reduce error *on average* below a district's own recent history. Results at earlier lead times, in shock seasons and per feature group are still to come.
- **Why it may have happened (hypotheses, untested):**
  - Most variance is between districts and is slow-moving, which the baseline already captures.
  - 120 features against about 600–1,000 training rows per fold invites overfitting; boosting with tuned depth is more prone to this than a bagged forest.
  - The labels are noisy crop-cut estimates.
  - The wet-zone cropland mask covers little paddy, so NDVI there is weak.
  - The hybrid model learns year-to-year residuals that may be mostly noise, then adds them on top of the baseline.

## Paired Wilcoxon vs baseline
*Source: as above; `RESULTS.md` §2.*
- **What it shows:** whether each model's absolute errors are systematically smaller than the baseline's on the same district-seasons.
- **How to read it:** p (Holm) < 0.05 means a real difference. The sign of r gives the direction, and its size gives the effect (|r| < 0.1 is negligible).
- **Result:** random forest, ridge, LightGBM and XGBoost all have p (Holm) ≈ 0.45 and |r| ≤ 0.065, so there is no detectable difference from the baseline. The hybrid is significantly worse (r = −0.19, p < 0.001).
- **What it does not prove:** that the models are *equal* to the baseline. A non-significant test is not evidence of equivalence. The rows are also not fully independent, because districts in the same season share weather.

## Conformal coverage and interval width
_pending (Task 7)_

## Earliness curve
_pending (Task 8)_

## Ablation and SHAP
_pending (Task 9)_

## Shock seasons
_pending (Task 10)_

## Per-district error map, yield-trend plots, validation scheme diagram
_pending_

## Why random k-fold would inflate scores here
*Source: as above; `RESULTS.md` §4.*
The same LightGBM model was scored on the same 808 rows two ways:
- **Forward-chaining:** R² 0.467, RMSE 639 kg/ha.
- **Random 5-fold:** R² 0.713, RMSE 468 kg/ha.

Random folds therefore make the model look about 27% better on RMSE and add about 0.25 to R². The reason is that random folds put *later* seasons, and other districts in the *same* season, into training. The model then learns that season's shared weather and fertiliser shock, and the yield trend after the test year: information that would not exist at forecast time. This is why random k-fold is banned for the main results (`random_kfold` raises an error).

## Limitations observed
_Filled from the data as work proceeds. Items known so far (Task 1, from the DCS files):_
- **Label noise:** DCS yield is a crop-cutting sample estimate, so the label itself is noisy.
- **Mahaweli H / Udawalawe merge:** until 2017/18 Maha and 2018 Yala these irrigation areas were reported separately; from then on they are inside their host districts. This structural break could not be measured cleanly (the 2016/17 drought falls just before it). Results will be reported (a) for all districts, (b) excluding post-merge rows of the affected districts, and (c) as error before vs after the merge. `mahaweli_merged` is not used as a model feature.
- **Missing district-seasons:** 101 of 2,375 are dropped. Jaffna has no Yala yield after 1983, and northern and eastern districts have gaps in the conflict years.
- **Raw-file defects:** the Production-Maha row order and two header typos were repaired in code. Because the production repair relied on yield × harvested area, production is not an independent check on yield.
- **Label availability:** the 6-month publication lag is an assumption, not a verified DCS release schedule.
- **Short feature history:** satellite features start in 2000, so the modelling table has about 25 seasons per district, not 47.
- **Paddy mask proxy (measured):** WorldCover 2021 class 40 at 250 m (≥50% cropland) covers 0.85× the DCS 2018–2024 mean Maha sown area nationally. In the dry zone it is 0.7–1.3×, but in the wet zone it covers only ~0.10–0.47× (Kegalle 0.10, Galle 0.14, Kalutara 0.18, Nuwara Eliya 0.22, Colombo 0.27, Kandy 0.37, Matara/Ratnapura 0.40, Gampaha 0.47). There, wet-zone paddy is often mapped as grassland or wetland, so NDVI rests on fewer pixels. Possible sensitivity test: add class 90 (herbaceous wetland) or lower the 50% threshold.
