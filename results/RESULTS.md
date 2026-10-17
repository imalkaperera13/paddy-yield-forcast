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
