# Results

Every number here was produced by code run on the real data. Source for this section:
- **Kaggle kernel** `imalkaperera/paddy-01-baselines-and-models`, **Version 2**, CPU, internet off, finished 2026-10-17 in 10,751 s.
- **Code commit** `3a1d358` (from the `COMMIT` file in the code dataset).
- **Data:** `paddy-master-table` Version 1, `master_table.parquet` SHA-256 `2a8fdc80…a158`, built 2026-10-17T11:32:55Z.
- **Raw outputs:** `results/tables/` (copied from the kernel output). Kernel run metadata: `results/runs_meta/01_models/`.

Setup: forward-chaining by season, lead 0 (forecast at the reference harvest date). 33 test seasons (2009/10 Maha – 2025/26 Maha), 808 district-seasons. Every model is scored on the same 808 rows, over 5 seeds (mean ± std).

## 1. Model comparison (`model_comparison.md`)

| model | RMSE (kg/ha) | MAE (kg/ha) | MAPE (%) | R² | RMSE (% of mean) |
|---|---|---|---|---|---|
| random_forest | 604.1 ± 2.9 | 435.0 ± 2.8 | 12.09 ± 0.05 | 0.523 ± 0.005 | 15.24 ± 0.07 |
| **baseline_last5** | 605.8 | 442.7 | 12.18 | 0.521 | 15.28 |
| ridge | 620.2 | 438.2 | 12.29 | 0.497 | 15.65 |
| xgboost | 633.1 ± 0.6 | 457.4 ± 1.4 | 12.54 ± 0.04 | 0.476 ± 0.001 | 15.97 ± 0.02 |
| lightgbm | 633.9 ± 4.9 | 457.7 ± 4.0 | 12.54 ± 0.08 | 0.475 ± 0.008 | 15.99 ± 0.12 |
| hybrid_lgbm | 661.1 ± 5.1 | 480.1 ± 3.1 | 13.27 ± 0.08 | 0.429 ± 0.009 | 16.68 ± 0.13 |

Ridge and the baseline are deterministic, so their std is 0.

**Note on ridge (620.2 here vs 598.7 in `results/predictions.csv`):** these are two different runs. 620.2 is this kernel (01): ridge fit on the whole training window of each fold. `results/predictions.csv` is a copy of the conformal predictions from kernel 02 (§8), where each model is fit **without** the last 3 training years, which are kept for calibration; there ridge scores 598.7. Both are on the same 808 rows. The RF difference (605.8 here vs 607.0 in predictions.csv) has the same cause. Details in `docs/decisions.md`.

## 2. Paired Wilcoxon test vs the baseline (`wilcoxon_vs_baseline.md`)
The test pairs each model's absolute errors with the baseline's on the same 808 rows. p-values are Holm-adjusted across 5 comparisons. A positive r means the model's errors are smaller.

| model | median abs. error (model / baseline) | share of rows better | rank-biserial r | p (Holm) |
|---|---|---|---|---|
| random_forest | 323.4 / 324.7 | 0.529 | +0.065 | 0.449 |
| ridge | 305.5 / 324.7 | 0.517 | +0.043 | 0.456 |
| lightgbm | 327.6 / 324.7 | 0.489 | −0.058 | 0.456 |
| xgboost | 321.2 / 324.7 | 0.489 | −0.056 | 0.456 |
| hybrid_lgbm | 339.4 / 324.7 | 0.460 | −0.186 | < 0.001 |

**Finding (failed hypothesis, reported plainly):** at lead 0, **no model is significantly better than the simple last-5-seasons mean**. The random forest is level with it (RMSE 604.1 vs 605.8; p = 0.45). The hybrid residual model is **significantly worse**.

## 3. Leave-district-out (`leave_district_out.csv`; seed 42, fixed parameters)
| model | RMSE | MAE | R² |
|---|---|---|---|
| baseline_last5 | 605.8 | 442.7 | 0.521 |
| ridge | 644.5 | 466.0 | 0.457 |
| lightgbm | 660.0 | 478.8 | 0.431 |

When the test district is removed from training, both models fall further behind the baseline. The baseline uses only that district's own history, so it is unaffected.

## 4. Random k-fold vs forward-chaining (`kfold_vs_forward_demo.csv`; LightGBM, fixed parameters, same 808 rows, 5 seeds)
| scheme | RMSE | MAE | R² |
|---|---|---|---|
| forward-chaining (honest) | 638.6 ± 3.1 | 460.4 ± 1.8 | 0.467 ± 0.005 |
| random 5-fold (leaky) | 468.5 ± 5.9 | 346.8 ± 4.4 | 0.713 ± 0.007 |

## 5. Sensitivity: training without the sanity-flagged rows (`sensitivity_sanity_flags.csv`; LightGBM, scored on the 768 unflagged test rows)
| variant | RMSE | MAE | R² |
|---|---|---|---|
| trained with flagged rows | 621.7 ± 4.4 | 446.9 ± 3.8 | 0.507 ± 0.007 |
| trained without flagged rows | 618.5 ± 2.9 | 443.3 ± 1.2 | 0.512 ± 0.005 |

The difference (≈3 kg/ha RMSE) is about the size of the seed-to-seed spread. Keeping the flagged rows does not change the conclusions.

## 6. Mahaweli merge and season diagnostics (`diagnostics_mahaweli_season_sanity.csv`)
RMSE, kg/ha:

| subset | n | baseline | ridge | RF | LightGBM | XGBoost | hybrid |
|---|---|---|---|---|---|---|---|
| (a) all rows | 808 | 605.8 | 620.2 | 604.1 | 633.9 | 633.1 | 661.1 |
| (b) excl. post-merge rows of affected districts | 740 | 592.2 | 604.9 | 589.9 | 615.3 | 616.0 | 645.3 |
| (c) affected districts, before merge | 64 | 361.1 | 376.0 | 395.7 | 425.9 | 419.3 | 388.4 |
| (c) affected districts, after merge | 68 | 737.4 | 767.9 | 741.0 | 809.2 | 796.4 | 813.3 |
| (c) other districts, before merge | 327 | 528.1 | 549.5 | 526.3 | 542.2 | 552.0 | 565.9 |
| (c) other districts, after merge | 349 | 677.0 | 683.3 | 670.4 | 702.9 | 697.7 | 745.2 |
| Maha | 424 | 645.0 | 663.6 | 644.2 | 664.6 | 660.9 | 695.0 |
| Yala | 384 | 559.2 | 568.6 | 556.4 | 598.2 | 601.0 | 621.6 |

In this table, RF = random forest and hybrid = hybrid_lgbm.

Errors rise after the merge in **both** affected and other districts. Most of the rise is therefore a period effect (the post-merge period includes the 2021–22 fertiliser-ban seasons), not a merge effect alone. In the affected districts the rise is larger: about 2.0× vs 1.3× for the baseline. That is consistent with an added break from the merge, but it rests on only 64 and 68 rows, so it is not conclusive.

## 7. Shock seasons (`shock_seasons.md`, figure `shock_bias_by_season.png`)
Computed locally by `python -m src.analysis.shocks` from the kernel-01 v2 predictions (seed-averaged); no new training. Bias = mean(predicted − observed); positive means over-forecast.

| shock group (seasons) | n | baseline RMSE / bias | RF RMSE / bias | ridge RMSE / bias |
|---|---|---|---|---|
| policy/economic, named (Maha 2021/22, Yala 2022, Maha 2022/23) | 74 | 1142.6 / +986.9 | 1170.2 / +923.3 | 1197.1 / +940.1 |
| weather, named (Maha 2010/11 floods) | 25 | 773.9 / +492.4 | 936.8 / +720.9 | 959.1 / +790.5 |
| data-chosen, cause unverified (Maha 2023/24, Yala 2023, Maha 2024/25) | 74 | 411.7 / +223.2 | 399.0 / +48.9 | 437.8 / +96.1 |
| all other test seasons | 635 | 521.0 / −162.4 | 499.1 / −109.0 | 513.8 / −78.1 |

- Every model over-forecasts the fertiliser-ban seasons badly: by about 1,400–1,560 kg/ha in Maha 2021/22, about half the observed mean of 2,853. The features carry no fertiliser information, so this is expected; it is a structural limit of weather- and satellite-based forecasting.
- **2010/11 Maha floods:** all models over-forecast *more* than the baseline (RF bias +721 vs +492). The rainfall and NDVI features did not warn of the flood loss at lead 0. Why is open: the flood came late in the season (Jan–Feb 2011) and may be poorly captured by 4-week district-mean rainfall; this is a hypothesis.
- **Recovery season (Maha 2022/23):** RF and ridge beat the baseline clearly (RMSE 389 and 344 vs 587). The baseline still averages in the ban year, whereas the models do not depend on it alone.
- Shock-season choice: the named seasons were fixed before looking at errors. The data-chosen seasons are the 3 largest other negative national yield anomalies vs the previous 10 same seasons (`shock_season_list.csv`).

## 8. Conformal prediction intervals (`conformal_coverage.md`, figures `conformal_reliability.png`, `conformal_coverage_by_season.png`)
- **Source:** Kaggle kernel `imalkaperera/paddy-02-conformal` v1, code and data commit `cd8d7da`, data `paddy-master-table` v2 (master SHA-256 `2a8fdc80…`). Seed 42, lead 0, the same 808 test rows.
- **Method:** split conformal. The calibration set is the last 3 harvest years of each training window; the model is tuned and fitted on the older years only.

Target: 90% intervals (α = 0.1).

| model | variant | coverage | mean width (kg/ha) | width as % of mean yield |
|---|---|---|---|---|
| baseline | standard | 0.873 | 1,901 | 47.9 |
| baseline | per-season | 0.875 | 1,944 | 49.0 |
| random forest | standard | 0.861 | 1,898 | 47.9 |
| random forest | per-season | 0.861 | 1,960 | 49.4 |
| ridge | standard | 0.877 | 2,097 | 52.9 |
| ridge | per-season | 0.873 | 2,082 | 52.5 |

**Coverage of the standard 90% intervals, split by the 4 shock seasons named in advance** (Maha 2010, Maha 2021, Yala 2022, Maha 2022):

| model | outside shocks (709 rows) | in shock seasons (99 rows) |
|---|---|---|
| baseline | 0.911 | 0.596 |
| random forest | 0.903 | 0.566 |
| ridge | 0.922 | 0.556 |

- **Overall:** coverage falls slightly short of 90% (86–88%). The whole shortfall comes from the shock seasons. Outside them, coverage is at or above the target.
- **Intervals are wide:** about ±950 kg/ha, roughly ±24% of mean yield. After the crisis seasons enter the calibration window, intervals widen further: mean width rises from about 1,500 kg/ha (test seasons up to 2019) to about 2,900–3,260 kg/ha (from 2022/23 Maha), and coverage there is 96–100%.
- **Per-season (Mondrian) calibration** did not improve overall coverage. It does balance the two seasons for the baseline: Maha rises from 0.85 to 0.87, and Yala falls from 0.90 to 0.88.
- **Point accuracy of the models fitted without the calibration years:** RMSE 607 (RF), 599 (ridge), 606 (baseline). These are close to kernel 01's values.

## 9. Feature-group ablation and SHAP (`ablation.md`, `ablation_wilcoxon.md`, `shap_group_importance.md`, figure `shap_top_features.png`)
- **Source:** Kaggle kernel `imalkaperera/paddy-04-ablation-shap` v1, commit `cd8d7da`, data `paddy-master-table` v2; 2,589 s.
- **Setup:** random forest, seed 42, lead 0, the same 808 test rows. Each variant is tuned inside each training window as in kernel 01.

| feature set | features | RMSE | MAE | R² |
|---|---|---|---|---|
| lagged yield + MODIS | 40 | 596.4 | 430.4 | 0.535 |
| all minus temperature/radiation | 54 | 600.7 | 432.1 | 0.529 |
| lagged yield only | 9 | 602.9 | 435.3 | 0.525 |
| **baseline (last-5 mean)** | 0 | 605.8 | 442.7 | 0.521 |
| all | 120 | 605.8 | 436.9 | 0.521 |
| all minus rainfall | 106 | 608.2 | 438.5 | 0.517 |
| lagged yield + rainfall | 23 | 608.6 | 438.2 | 0.516 |
| all minus MODIS | 89 | 610.1 | 439.4 | 0.514 |
| lagged yield + temperature/radiation | 75 | 611.5 | 440.0 | 0.512 |
| no lagged yield (satellite + climate only) | 111 | 682.5 | 527.0 | 0.391 |

Paired Wilcoxon tests, Holm-adjusted over all 17 comparisons:
- **Removing MODIS significantly worsens the full model:** r = −0.12, p (Holm) = 0.037. No other group has a significant effect.
- **The best variant, lagged yield + MODIS, still does not significantly beat the baseline:** r = +0.10, p (Holm) = 0.25. Its raw p is 0.019, so the gain may be real but small.
- **Dropping the lagged yields** makes the model much worse than both the baseline and 'all' (r = −0.26 and −0.30; p (Holm) < 1e-8).

**SHAP**, share of total mean |SHAP|: lagged yield 76.6% (9 features), temperature/radiation 13.0% (66 features), MODIS 7.0% (31), rainfall 3.4% (14). The top features are the last-5 mean (230 kg/ha mean |SHAP|), the trend (134), the previous same-season yield (106) and the long-term mean (84). The strongest non-yield feature is pre-season rainfall (7 kg/ha).

**Reading:** almost all of the predictive skill comes from each district's own yield history. MODIS vegetation is the only satellite/climate group with a measurable, but small, contribution. Rainfall and temperature/radiation add noise rather than skill at lead 0.

## 10. Earliness and growth-window shift (`earliness.md`, `earliness_wilcoxon_vs_baseline.md`, `window_shift_sensitivity.md`, figure `earliness_curve.png`)
- **Source:** Kaggle kernel `imalkaperera/paddy-03-earliness` v1, commit `cd8d7da`, data `paddy-master-table` v2; 4,581 s.
- **Setup:** seed 42. Each lead uses its own feature table (only data available that many weeks before the reference harvest) and its own forward-chaining folds. All leads are scored on the same 808 rows.

RMSE in kg/ha, R² in brackets:

| weeks before harvest | baseline | random forest | ridge |
|---|---|---|---|
| 12 | 605.8 (0.521) | 614.9 (0.506) | 625.5 (0.489) |
| 10 | 605.8 (0.521) | 612.6 (0.510) | 639.6 (0.466) |
| 8 | 605.8 (0.521) | 613.3 (0.509) | 634.9 (0.473) |
| 6 | 605.8 (0.521) | 612.4 (0.510) | 647.3 (0.453) |
| 4 | 605.8 (0.521) | 608.4 (0.517) | 651.2 (0.446) |
| 2 | 605.8 (0.521) | 609.9 (0.514) | 610.1 (0.514) |
| 0 | 605.8 (0.521) | 605.8 (0.521) | 620.2 (0.497) |

- **The baseline does not depend on lead:** the same-season yields it uses are already published 12 weeks before harvest.
- **Random forest** improves only slightly as harvest approaches (615 → 606 kg/ha) and never beats the baseline. By Wilcoxon (Holm) it is never significantly different from the baseline: p (Holm) = 1.0 at every lead.
- **Ridge** is unstable and worse at most leads. At lead 10 its errors are significantly larger than the baseline's (r = −0.14, p (Holm) = 0.006).
- **Consistency check:** the lead-0 random forest and ridge reproduce kernel 01's seed-42 values exactly (605.8 and 620.2), and kernel 04's 'all' variant (605.8).

**Window shift** (lead 0; growth windows moved ±2 weeks):

| shift | baseline | random forest | ridge |
|---|---|---|---|
| −2 weeks | 605.8 | 600.3 | 638.1 |
| 0 | 605.8 | 605.8 | 620.2 |
| +2 weeks | 605.8 | 604.4 | 630.6 |

The random forest changes by at most 5.5 kg/ha, within run-to-run variation. Conclusions do not depend on the nominal planting dates.

## 11. Static paddy mask vs WorldCover cropland: 7 study districts, leads 12 / 8 / 4 weeks (Step 6)
- **Sources:**
  - Kaggle kernels `imalkaperera/paddy-06a-masked-models` v1 (forward chaining + leave-district-out; 180 runs, 18,148 s of fitting) and `imalkaperera/paddy-06b-masked-followups` v1 (conformal, ablation, SHAP). Both private, CPU, internet off.
  - Code commit `272c570`; scikit-learn 1.6.1, LightGBM 4.6.0, XGBoost 3.4.1, SHAP 0.54.1.
  - Data: private dataset `imalkaperera/paddycast-masked` (the version that includes `feature_list.json`), holding `master_masked.parquet` and `master_unmasked_7d.parquet`.
  - Tables: `python -m src.experiments.masked_compare report` → `results/tables/masked_*.csv`.
- **Setup:**
  - 7 districts (Anuradhapura, Ampara, Kurunegala, Polonnaruwa, Batticaloa, Hambantota, Trincomalee).
  - Forward chaining at each lead with its own label-availability rule: 33 test seasons (Maha 2009/10 – Maha 2025/26), 231 district-seasons, the same rows for every mask × lead × model × seed. 5 seeds.
  - **Masked** = features inside the static 2000–2008 paddy mask (Steps 2–3). **Unmasked** = the pilot's WorldCover-cropland features for the same rows. Same folds, seeds and tuning grids; only the feature values differ.
  - The baseline (last-5 same-season mean) does not depend on the mask or the lead.
- **Not comparable with §1–10:** those use 25 districts and 808 rows. The baseline RMSE here (621.6) is on different rows from the pilot's 605.8.

### 11.1 Accuracy (`masked_model_comparison.csv`)
RMSE in kg/ha, mean over 5 seeds (seed std ≤ 6.5; ridge and baseline have std 0):

| model | masked L12 | masked L8 | masked L4 | unmasked L12 | unmasked L8 | unmasked L4 |
|---|---|---|---|---|---|---|
| **baseline_last5** | 621.6 | 621.6 | 621.6 | 621.6 | 621.6 | 621.6 |
| random_forest | 647.0 | 643.9 | 635.5 | 648.2 | 644.1 | 636.0 |
| ridge | 650.5 | 722.0 | 660.5 | 661.9 | 696.1 | 662.2 |
| xgboost | 698.2 | 683.3 | 652.1 | 703.1 | 687.2 | 654.8 |
| lightgbm | 691.8 | 680.3 | 656.4 | 699.2 | 685.1 | 657.7 |

R²: baseline 0.492; best model, random forest masked L4, 0.469.

### 11.2 Masked vs unmasked (`masked_vs_unmasked_wilcoxon.csv`, `masked_vs_unmasked_ldo_wilcoxon.csv`)
Paired Wilcoxon on per-row absolute errors (seed-averaged), 231 pairs. Holm across 4 models × 3 leads = 12 tests. Positive r means the masked features give smaller errors.
- **Forward chaining: no significant difference** for any model at any lead.
  - Smallest adjusted p: ridge at lead 12, r = +0.22, raw p = 0.004, p (Holm) = 0.0505.
  - Random forest: |r| ≤ 0.025 and raw p ≥ 0.74 at every lead; RMSE differs by at most 1.2 kg/ha.
- **Leave-district-out (fixed parameters, seed 42):** one significant difference, and it favours the unmasked features: random forest at lead 8, r = −0.22, p (Holm) = 0.047 (RMSE 709.3 masked vs 702.8 unmasked). No other model or lead differs.
- **Finding (failed hypothesis, reported plainly):** restricting MODIS, CHIRPS and ERA5-Land features to the static paddy mask **does not improve** forecasts in these 7 districts at 4–12 weeks lead.

### 11.3 Models vs the baseline (`masked_wilcoxon_vs_baseline.csv`; Holm over 24 tests)
- **No model is significantly better than the baseline** at any lead with either mask.
- **Ridge at lead 8 is significantly worse**, with both masks:
  - masked: r = −0.27, p (Holm) = 0.008;
  - unmasked: r = −0.24, p (Holm) = 0.032.
- **The random forest has a slightly smaller median error than the baseline** (median gain +3.4 to +10.5 kg/ha; r +0.02 to +0.08; p (Holm) = 1.0) but a **higher RMSE**, so it makes larger misses.

### 11.4 Earliness (`masked_earliness_wilcoxon.csv`; lead 12 and 8 vs lead 4, Holm over 20 tests)
- **Errors grow when the forecast is made earlier.** The difference is significant for:
  - XGBoost at leads 8 and 12 with both masks (r −0.23 to −0.26);
  - ridge at lead 8 with both masks;
  - LightGBM at lead 8, unmasked.
- **The random forest is the most stable model:** 635.5 → 647.0 kg/ha from lead 4 to lead 12 (masked); no significant difference (smallest p (Holm) = 0.076).
- **The baseline is the same at every lead:** the same-season yields it uses are already published 12 weeks before harvest.

### 11.5 Leave-district-out (`masked_leave_district_out.csv`)
- **Every model falls well behind the baseline** when the test district is left out of training:
  - random forest 695.1–709.3 kg/ha;
  - ridge 699.2–764.0;
  - LightGBM 714.4–760.4;
  - XGBoost 728.1–781.5;
  - baseline 621.6.

### 11.6 Conformal intervals (`masked_conformal.csv`, `masked_conformal_by_season.csv`; seed 42; target 90%)
- **Coverage is below 90% everywhere.**
  - Standard: 0.831–0.879 across baseline, ridge and random forest, both masks and all leads.
  - Mondrian by season: 0.814–0.853.
- **Intervals are wide:** mean width 2,034–2,360 kg/ha, i.e. 45–53% of mean yield.
- **The Mondrian variant matches the two seasons' error levels:** Yala intervals are narrower (1,613–1,893 kg/ha) and Maha intervals wider (2,430–2,643).
  - Coverage: Maha 0.798–0.857, Yala 0.812–0.875.
  - Standard intervals by season: Maha 0.765–0.857, Yala 0.857–0.911.
- **No infinite intervals:** every fold calibrates on 21 rows of the test row's own season.
- **The mask changes coverage by at most 5 of 231 rows** for any model, variant and lead.

### 11.7 Feature-group ablation and SHAP (random forest, seed 42; `masked_ablation*.csv`, `masked_shap_group_share.csv`)
- **No feature set beats the baseline.** The best variant at each mask × lead scores 632.9–640.0 kg/ha, against the baseline's 621.6.
- **Lagged yield only** gives 640.0 at every lead with both masks. It uses no satellite data, so it is identical in both.
- **Significant results after Holm (within each mask × lead):**
  - Dropping the lagged yields is significantly worse than both the baseline and 'all' in every case (r −0.34 to −0.43).
  - At unmasked lead 12, lagged yield + rainfall beats 'all' (r +0.23, p (Holm) = 0.042).
  - Nothing else is significant. Removing MODIS is not significant here, unlike the 25-district lead-0 pilot (§9).
- **SHAP share** of total mean |SHAP|:

  | feature group | share |
  |---|---|
  | lagged yield | 74.6–81.5% |
  | temperature/radiation | 7.3–12.2% |
  | MODIS | 6.3–8.2% |
  | rainfall | 4.6–5.5% |

  The lagged-yield share grows with lead (from 75–76% at lead 4 to 81–82% at lead 12), as fewer satellite composites are available before the cutoff.

### 11.8 Reading
- **Main finding:** in the 7 flat dry/intermediate-zone districts, at 4–12 weeks before harvest:
  - no model beats the last-5-seasons mean;
  - the static paddy mask gives no measurable gain over WorldCover cropland;
  - skill still comes almost entirely from each district's own yield history.
- **Possible reasons, not tested:**
  - Only 231 test rows, so small effects cannot be detected.
  - The features are district means, which smooth out field-level signal.
  - The static mask is conservative and falls below DCS sown area after 2008 (e.g. Anuradhapura Yala ratio 1.25 → 0.52; `mask_static_validation.csv`).
  - The 2021–22 fertiliser-ban seasons dominate the errors, and no feature can see them.

