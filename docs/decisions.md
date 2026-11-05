# Decisions log

Every modelling and data choice, with the reason. Newest entries at the bottom of each section.

## Task 0 — scaffold (2026-10-17)
- **Python**: 3.11 was the target. The local machine uses **3.12**, because only 3.12 and 3.14 are installed and 3.14 is too new for the pinned stack. Every pinned package supports 3.12, and Kaggle images are 3.11/3.12. Each kernel logs its exact version.
- **Python target 3.11 (original)**, dependencies pinned in `requirements.txt`. The scaffold was built and tested on Python 3.13 in a Linux container; the real runs use 3.11 on Windows or Kaggle.
- **Config** lives only in `configs/config.yaml`. Code reads it through `src/config.py`.
- **Seeds**: base seed 42; a spread is reported over the five seeds in `project.seeds_for_spread`.
- **Tracking**: MLflow with a local file store (`./mlruns`, git-ignored).
- **Raw data** is never committed and never edited. Cleaning happens only in scripts.
- **No random k-fold** anywhere (`validation.random_kfold_allowed: false`, checked in tests).

## Task 0b — addendum: Kaggle workflow (2026-10-17)
- **Data sources**: official sources only (DCS, HDX, Earth Engine, NASA POWER, CHIRPS). Kaggle is not a data source. The only things uploaded to Kaggle are our own processed table and our code.
- **Training on Kaggle** (not Colab). There are two private datasets:
  - `paddy-master-table`: master table, data dictionary, and `manifest.json` (sha256, build time, git commit).
  - `paddy-yield-code`: snapshot of `src/` and `configs/`, plus a `COMMIT` file.
  The code comes in as a dataset rather than a git clone, so kernels need no internet and no stored token.
- **One kernel per experiment group**: models, conformal, earliness, ablation+SHAP, shock. Defaults are `is_private: true`, `enable_gpu: false`, `enable_internet: false`, and a test enforces them.
- **Kaggle limits** (checked 2026-10-17, see kaggle/README.md): 12 h session, about 20 GB in `/kaggle/working`, about 30 h/week GPU (not used). Each run is split into resumable units, one per fold or seed.
- **Library versions on Kaggle** are the image defaults, since installs need internet. Each kernel logs them to `run_info.json`, and the report quotes those versions.
- **Dataset and kernel versions used for reported results**: _recorded here after each upload_

| date | dataset | version | git commit | note |
|---|---|---|---|---|

## Task 1 — DCS yields (2026-10-17)
Raw files: the 8 district series (1979–2025) and the 2022/23 Maha metric PDF used as a cross-check. They were downloaded by hand on 2026-10-17; hashes are in `docs/raw_provenance.md`. Code: `src/data/dcs.py` and `src/data/build_yield.py`.

**Season-year convention (APPROVED 2026-10-17).** Maha "2021/2022" is labelled `season_year = 2021`, its start (planting) year, with `harvest_year = 2022`. Yala 2022 is labelled `season_year = harvest_year = 2022`. `season_order = 2*harvest_year + (0 for Maha, 1 for Yala)` sorts seasons by harvest time: Maha 2021/22 comes before Yala 2022, which comes before Maha 2022/23. Forward-chaining splits use `season_order`, never `season_year` alone.

**Mahaweli H / Udawalawe (APPROVED 2026-10-17).** DCS reports these irrigation areas as separate rows until 2017/18 Maha and 2018 Yala. From then on they are included in their host districts (footnote in every file). Proposal: model the **25 administrative districts exactly as published**, leave out the two special-area rows (they have no admin polygon), and add a `mahaweli_merged` flag (0 before the merge, 1 from it). **The flag is NOT a model feature**, because it is almost the same as a time dummy. It is used only for diagnostics: results are reported (a) for all districts, (b) excluding the post-merge rows of the affected districts, and (c) as error before vs after the merge date. The break could not be measured cleanly because the 2016/17 drought falls in the "before" window, so it is listed as a limitation.

**Units.** Yield is kg per NET hectare harvested. Extents are GROSS hectares. Production is in thousand tonnes in the raw files and stored as tonnes (`production_t`). Because yield is per net hectare and extent is gross, production / (yield × harvested) is normally 0.78–1.00 (5th–95th percentile), not exactly 1.

**Raw-file defects found and repaired in code (raw files untouched).**
1. The header "2000/2021" should be 2020/2021 in the Maha yield, sown and harvested files. Repaired by sequence.
2. The header "198" should be 1980 in Production-Yala. Repaired by sequence.
3. In Production-Maha, for 1978/79–1982/83 and 2008/09–2024/25 the values are in the standard district order but the row labels are in a different order. For each season, the order that agrees with yield × harvested area is kept. Verified against the PDF: Ampara 318,520 t vs 318,517 t; Hambantota 178,140 t vs 178,137 t.
4. A published yield of 0 is a "not reported" placeholder (for example, Ampara 1990 Yala: yield 0 but 125,100 t produced). 21 rows (1985/86 Maha and the 1988–1994 Yala seasons) are treated as missing.

**Dropped rows (no imputation).** 101 of 2,375 district-season rows have no yield: 22 Maha and 79 Yala. That leaves 2,274 rows (Maha 1,178 rows for 1978/79–2025/26; Yala 1,096 rows for 1979–2025). Jaffna has no Yala yield after 1983 (42 missing). The other gaps are northern and eastern districts in the conflict years. Per-district counts: `results/tables/dcs_missing_counts.csv`; figure: `results/figures/dcs_missing_heatmap.png`.

**Sanity flags (kept, not removed).** See `results/tables/dcs_sanity_flags.csv`:
- 103 rows where production / (yield × harvested) falls outside 0.6–1.1. This is mostly Nuwara Eliya at a stable ~0.56, which looks like a net-vs-gross extent difference.
- 11 robust outliers (|z| > 3.5).
- 2 rows with harvested > sown (Puttalam 1983/84 Maha: 19,763 > 10,559, a likely raw error).

National check: the area-weighted mean of the reporting units matches the published Sri Lanka yield to a median of 0.2%. The exceptions are 2010/11 Maha (23%, the January–February 2011 floods, with the special areas unreported that season) and 1990 Yala (18%).

**Label-availability assumption (APPROVED; ASSUMPTION, NOT A VERIFIED FACT).** We do not know DCS's actual release dates. We assume a season's yield becomes public `data.label_availability_lag_months` = **6 months** after that season's reference harvest date. A past label may be a feature only if `harvest_ref(label season) + 6 months <= forecast date`. In practice, with forecasts made near harvest:
- the same season of the previous year is always available;
- the immediately preceding (opposite) season is usually NOT available, and the code excludes it automatically.

The defaults are the same-season previous-year yield, the long-term same-season mean, and the last-5-same-season mean (the baseline). A sensitivity run with other lags is planned.

**Leakage rules for all later steps (APPROVED).**
- Target-season `production_t`, `harvested_ha` and `sown_ha` are never features. Only previous seasons' values may be used, and only if available before the forecast date under the same lag rule. `configs/config.yaml: forbidden_features` is enforced by tests.
- The modelling table starts where satellite and climate features exist: 2000/01 Maha and 2000 Yala (MODIS begins 2000-02-18). Yields from 1979–1999 are used only for lags and baselines. This is stated in the data card.
- The sanity-flagged rows are kept: 116 flags on 114 rows (103 production-ratio, 11 robust-outlier, 2 harvested>sown). A sensitivity run is reported with and without them (`sanity_flag` column).
- Shock tests: 2010/11 Maha (January–February 2011 floods) is a candidate. Other shock seasons are chosen from the data (largest negative area-weighted national yield anomalies vs the past-only mean), not assumed.
- Methods note: the Production-Maha row-order repair was decided using yield × harvested area, so any production-derived quantity is **not independent of yield** and must not be used to validate yield.

**District names → boundaries.** All 25 DCS districts map to HDX COD-AB v03 adm2 (P-codes LK11–LK92); see `configs/district_map.csv`.

## Task 2 — features (2026-10-17, MODIS + CHIRPS first)
**Season calendar and growth-stage windows** (`configs/config.yaml: features`). Each season is anchored at a nominal planting date: Maha 15 October (of `season_year`), Yala 15 April. From the anchor come 5 consecutive 4-week blocks (`b1`–`b5`, 140 days, covering typical 3.5–4.5-month varieties) and 1 pre-season block (`pre1`, the 4 weeks before the anchor, for land preparation and antecedent rain). The reference harvest date is anchor + 140 days: 4 March for Maha, 2 September for Yala. Sensitivity: windows shifted by −2 and +2 weeks (`--shift`), with the forecast date unchanged. This is a single national calendar; in reality planting dates vary by district and by irrigation versus rainfed land (limitation).

**Forecast date and earliness.** forecast_date = harvest_ref − lead_weeks. Only data available at that date is used:
- a CHIRPS pentad counts if its end + 5 days ≤ forecast date;
- a MODIS composite counts if its start + 16 days + 5 days ≤ forecast date.

The 5-day latencies are ASSUMPTIONS (`features.data_latency_days`). The features use the FINAL CHIRPS product, which a real-time system would only have about 3 weeks later (limitation).

**Label lags at the forecast date** (6-month assumed publication lag). The same season of the previous year is always available. Because Yala's reference harvest (2 September) + 6 months = 2 March falls just before Maha's (4 March), the previous Yala becomes available for a **lead-0 Maha forecast only** (by 2 days), and not for any lead of 2 weeks or more. This follows mechanically from the assumed lag and is checked by a test; it is noted because the earliness curve may show a small step between lead 0 and lead 2 for Maha.

**Extraction** (`src/data/ee_extract.py`, run on the local machine):
- District polygons: HDX COD-AB adm2, simplified by 100 m in UTM 44N.
- Cropland mask: WorldCover v200 (2021) class 40, aggregated to the MODIS 250 m grid; a pixel is cropland if ≥ 50% of it is class 40. This is a proxy for paddy, with no paddy class and a 2021 map applied to all years (limitation).
- MODIS MOD13Q1 v061: NDVI and EVI × 0.0001, SummaryQA ≤ 1, cropland-masked district mean and valid-pixel count per 16-day composite.
- CHIRPS pentad: unmasked district mean, from 1981, so past-only rain anomalies have a long history.
- Method: tables are pulled per year with getInfo() into `data/interim/ee/`, which is resumable. Export.table.toDrive remains the fallback.

**Season features** (`src/data/features.py`):
- rain per block and season total;
- NDVI and EVI mean per block, season max, and the sum over composites (`ndvi_auc`);
- MODIS valid-pixel fraction per block;
- an anomaly for each climate feature vs the same district-season in previous years only (≥ 3 years);
- yield lags: previous same season, mean of all available same-season yields, last-5 mean (the baseline), 10-season linear trend, and last available label plus its age. These are all filtered by the 6-month lag rule. The previous same-season sown extent and harvested/sown ratio are also included.

**Shock candidates from the data** (national area-weighted anomaly vs the previous 10 same seasons): 2021/22 Maha −35.4%, 2022 Yala −23.4%, 2022/23 Maha −11.8% (the fertiliser-ban period, a policy shock), and 2010/11 Maha −8.3% (the floods). Drought and flood shock seasons for Task 10 will be selected excluding the ban seasons.

## Task D — validation (2026-10-17)
Code: `src/validation/splitters.py`; tests: `tests/test_splitters.py`, run on a SYNTHETIC grid and on the real master table at leads 0, 6 and 12 weeks.
- **(a) Forward-chaining by season.** Each fold tests ONE season (all districts). Training rows are those whose label would have been published by that season's forecast date (reference harvest + 6-month assumed lag), and they must be earlier in `season_order`. **First test season: 2009/10 Maha**, the first season whose available training labels span ≥ 10 harvest years (2000–2009). This gives 33 folds (2009/10 Maha to 2025/26 Maha) and 808 test district-seasons. Ten years is the brief's minimum. Starting earlier would leave the first models with too few seasons; starting later would waste test seasons.
- **(b) Leave-district-out.** The same folds, with the test district removed from training entirely, so the test measures generalisation to an unseen district while still respecting time. 808 folds.
- **(c) Calibration split** (for split conformal). Inside each training window, the most recent 3 harvest years are the calibration set and the rest is the proper training set.
- **Hyper-parameter tuning.** Inner expanding-window splits over the last 3 harvest years of each training window (`inner_time_splits`); test seasons are never used.
- **Random k-fold** raises an error in the main code. It is used only in the documented demonstration of why it inflates scores.
- Fold list: `results/tables/validation_folds_lead0.csv`.

## Task 2b — ERA5-Land (2026-10-17)
- `ECMWF/ERA5_LAND/DAILY_AGGR`, 1991–2026, district means at 11,132 m. It is pulled per quarter because a year has 25 × 365 rows, more than the 5,000-row limit per request. 37 yearly files, no failures.
- Only total evaporation is used, sign-flipped. The component evaporation bands are mislabelled in the source (EE catalog note).
- ASSUMPTION: 7-day data latency (ERA5T is ~5 days behind real time). Heat-stress day = daily max ≥ 35 °C (config).
- Result: the master table has 120 features. Because 120 features is about a tenth of the 1,266 rows, regularised and tree models are preferred, and the feature-group ablation (Task 9) will show whether this group helps.

## Task E1 — kernel 01 models run (2026-10-17)
- **Kernel** `imalkaperera/paddy-01-baselines-and-models` v2 (CPU, internet off), 10,751 s. Inputs: `paddy-master-table` v1 (master SHA-256 `2a8fdc80…`) and `paddy-yield-code` v1 (COMMIT `3a1d358`). The kernel wrapper came from the next commit; it differs only in how it finds the datasets.
- **v1 failed** after 10 s: the code dataset's zip is auto-extracted *without* the `repo/` folder, so the `repo/COMMIT` search failed. Fixed by finding `COMMIT` next to `src/` with `os.walk(followlinks=True)` and printing `/kaggle/input` at start.
- **Library versions differ from `requirements.txt`.** Kaggle's image provides sklearn 1.6.1, lightgbm 4.6.0, xgboost 3.4.1, numpy 2.1.3, pandas 2.3.3 and Python 3.13.15; the run used these, not the pinned versions. The exact versions are recorded in `results/runs_meta/01_models/run_info.json`. MLflow is not installed on the image, so tracking fell back to `runs_log.jsonl` (by design).
- **Result:** no model beats the last-5 baseline at lead 0 (Wilcoxon, Holm). The hybrid is significantly worse. Random 5-fold inflates R² from 0.47 to 0.71. Details: `results/RESULTS.md`.
- **Decision:** keep the baseline as the reference. The next steps test where models *could* add value: earliness (is there signal before harvest?), shock seasons, feature-group ablation, and conformal intervals around the baseline and the random forest.

## Tasks E2–E5 — follow-up experiments (2026-10-18)
- **Scope, to keep compute bounded:** kernels 02–04 use the baseline, ridge and random forest (the best model in kernel 01), with one seed (42; `followup_seed`). Kernel 01 showed the seed spread is small (RF RMSE std 2.9 kg/ha).
- **E2 conformal (kernel 02):** split conformal. Per fold, calibration = the last 3 harvest years of the training window; the model is tuned and fitted on the older years only. q = the ceil((n+1)(1−α))-th smallest calibration |residual|. Two variants: standard, and Mondrian by season (separate q for Maha/Yala). Reliability at α ∈ {0.5, 0.4, 0.3, 0.2, 0.1, 0.05}. Point accuracy here is lower than kernel 01's because the models see 3 fewer years; this is reported separately.
- **E3 earliness + window shift (kernel 03):** feature tables for leads 2–12 and shifts ±2 weeks were built with `python -m src.data.build_master --lead L` / `--shift S`. They were built in the Claude workspace from the same interim inputs: the lead-0 rebuild reproduces `master_table.parquet` exactly (SHA-256 `2a8fdc80…`). They are uploaded flat in `paddy-master-table` v2 (per-file SHA-256 in `manifest.json`). Each lead uses its own folds (labels published by that forecast date), and all leads are compared on the same rows.
- **E4 ablation + SHAP (kernel 04):** RF on 9 feature sets (all; lagged only; lagged + each group; all minus each group; no lagged). Paired Wilcoxon vs the baseline and vs 'all', Holm-adjusted. SHAP: TreeExplainer on each fold's own model (refitted with that fold's chosen parameters) and that fold's test rows only.
- **E5 shocks:** no training is needed, so the analysis runs locally from saved predictions (`src/analysis/shocks.py`); kernel 05 is not used. Shock seasons: named in advance (Maha 2021, Yala 2022, Maha 2022 policy/economic; Maha 2010 floods), plus the 3 largest other negative national anomalies from the data (Maha 2023, Maha 2024, Yala 2023; cause unverified).

## Task E2 — kernel 02 run (2026-10-18)
- Kernel `imalkaperera/paddy-02-conformal` v1, CPU, internet off. Inputs: `paddy-master-table` v2 and `paddy-yield-code` v2 (COMMIT `cd8d7da`). The run log ends with "experiment 02 complete". Outputs are in `results/tables/conformal_*`, `predictions_conformal.csv` and `results/figures/conformal_*.png`; run metadata is in `results/runs_meta/02_conformal/`.
- Result summary in `results/RESULTS.md` §8. The shock-split coverage (90–92% outside, 56–60% inside the 4 named shock seasons) was computed locally from `predictions_conformal.csv`. The shock seasons are the ones fixed in `configs/config.yaml` before any interval was seen.

## Task E4 — kernel 04 run (2026-10-18)
- Kernel `imalkaperera/paddy-04-ablation-shap` v1 (CPU, internet off), 2,589 s, code COMMIT `cd8d7da`, data v2. The log ends with "experiment 04 complete". Outputs: `results/tables/ablation*`, `shap_*`; `results/runs_meta/04_ablation_shap/`.
- Fix after the run (presentation only, no numbers changed):
  - The SHAP figure legend showed default colours, so it was redrawn locally from `shap_feature_importance.csv` with `redraw_shap_figure`.
  - `ablation.md` printed R² with 1 decimal, so it was regenerated from `ablation.csv` after `to_markdown` gained 3-decimal formatting for R²-type columns.

## Task E3 — kernel 03 run (2026-10-18)
- Kernel `imalkaperera/paddy-03-earliness` v1 (CPU, internet off), 4,581 s, code COMMIT `cd8d7da`, data v2. The log ends with "experiment 03 complete".
- The log is large (8.9 MB) because of 8,034 repeated scikit-learn warnings: "Skipping features without any observed values" (`SimpleImputer`). At long leads, the late-season blocks (b4, b5) do not exist yet, so those columns are entirely empty and the imputer drops them. This is expected behaviour, not an error.
- The `.md` tables were regenerated locally from the `.csv` files with 3-decimal R² (presentation only).

## Step F — report outputs and reproducibility (2026-10-18)
- `python -m src.run_all` rebuilds, from the raw DCS files and saved Earth Engine extracts:
  - the tidy yields;
  - the master table and its 8 lead/shift variants;
  - the descriptive figures, the shock analysis, the report figures and `results/predictions.csv`.

  A run on 2026-10-18 reproduced the master table and all variants with identical SHA-256. Model training runs on Kaggle (kernels 01–05); `--experiments` reruns the same code locally.
- `results/predictions.csv` = kernel-02 conformal predictions (baseline, ridge, RF; seed 42), with `lo`/`hi` from the standard split-conformal q at α = 0.1. Kernel-01 point predictions for all models and seeds are in `results/tables/predictions_models.csv`.
- **New figures:**
  - `validation_scheme.png`, drawn from `validation_folds_lead0.csv`;
  - `district_error_map.png`, per-district RMSE of the seed-averaged kernel-01 predictions, with the HDX boundaries simplified for Earth Engine.
- Kernel 05 is now `paddy-05-final-model` (final fits). The old shock-test kernel stub was removed because shock analysis needs no training.

## Task E5 — kernel 05 final models (2026-10-18)
- Kernel `imalkaperera/paddy-05-final-model` v1 (CPU, internet off), 74 s, code COMMIT `868c55f` (code dataset v3), data v2. The log ends with "experiment 05 complete".
- Both random forests were tuned to `min_samples_leaf` 15 and `max_features` 0.5, and trained on 1,266 rows. Conformal q (α = 0.1, calibration harvest years 2024–2026, 123 rows): 910.6 kg/ha (all features) and 817.5 kg/ha (lagged + MODIS).
- `models/final_models.json` and `models/model_card.md` are in git. The `.joblib` files are excluded by `.gitignore`; they are in the kernel output and `results/kaggle_runs/05_final_model/` on the PC.

## Step 1 — study districts (2026-11-05)
- **Why:** supervisor feedback (27 Oct 2026). Small, terraced fields in hilly areas give mixed satellite signals, so the project starts with large, flat paddy districts.
- **Inputs:** DCS sown extent (`data/interim/dcs_yield_tidy.csv`, as published) and SRTM 30 m slope over ESA WorldCover v200 cropland (`scripts/district_slope.py`, 90 m, run 2026-11-05 → `data/interim/ee/district_slope.csv`).
- **Rules** (`scripts/select_districts.py`):
  - mean annual sown extent (Maha + Yala) over harvest years 2016–2025 **≥ 50,000 ha**;
  - mean cropland slope ≤ 3° and share of cropland steeper than 5° ≤ 0.10;
  - dry or intermediate zone.
- **Selected (7):** Anuradhapura, Ampara, Kurunegala, Polonnaruwa, Batticaloa, Hambantota, Trincomalee → `data/processed/selected_districts.csv`; all 25 districts with every criterion in `results/tables/district_screening.csv`.
- **Kurunegala is intermediate zone, not dry zone.** It is kept because it passes the area and flatness rules (126,157 ha/year; mean cropland slope 1.87°), and the user confirmed it.
- **Monaragala** (dry zone, flat: 1.79°) is excluded only because its mean annual sown extent is **49,176 ha**, 824 ha below the 50,000 ha cutoff. It is the first district to add if the study is widened.
- ASSUMPTION: the dry-zone list in `select_districts.py` comes from general agro-climatic zoning and was not checked against an official zone map in this repo.
- Anuradhapura and Hambantota are in `mahaweli_affected_districts`; their DCS rows follow the pilot's Mahaweli handling, which should be kept in mind when comparing mask area with DCS sown extent.
- The pilot models (25 districts) are unchanged; later steps compare results on the 7 districts.

## Step 2 — seasonal active paddy mask (2026-11-05)
- **Code:** `src/data/paddy_mask.py` (Earth Engine, one request per season for the 7 districts, resumable, retry on timeout) and `scripts/mask_validation.py` (offline comparison with DCS).
- **Method:** a simplified version of Xiao et al. (2005) on MODIS MOD09A1 v061 (8-day surface reflectance, 500 m), because MOD13Q1 has no 1.6 µm SWIR band for LSWI. The mask is evaluated on the MODIS 250 m (MOD13Q1) grid, so each 500 m index value covers four 250 m cells. Clear observations only (StateQA cloud state clear, no shadow). Inside WorldCover class-40 cropland (≥ 50% of the 250 m pixel, the pilot's `cropland_fraction`). Per season, a pixel is active paddy if it shows:
  1. **flooding** in the flood window (anchor −30 to +75 days): at least one observation with LSWI + 0.05 ≥ EVI and EVI < 0.35;
  2. **green-up** in the grow window (+30 to +140 days): max EVI ≥ 0.35 and max EVI − flood-window min EVI ≥ 0.20;
  3. **harvest drop** in the harvest window (+100 to +180 days): grow max EVI − harvest-window min EVI ≥ 0.20.
- **Variants:** *strict* applies all three rules; *relaxed* also accepts rule 1 where the pixel has no clear flood-window observation (Maha is often cloudy at transplanting). `flood_clear_share` records how much of the cropland had at least one clear flood-window observation.
- **No tuning on DCS:** windows and thresholds are in `configs/config.yaml` (`features.paddy_mask`) and were fixed before any comparison with DCS, so the DCS check is independent. Any later change must be logged here with the reason.
- **Validation:** mask area is compared with DCS sown extent **per season** (Maha and Yala separately), never with the annual Maha + Yala total. Metrics per district × season: Pearson and Spearman r across years, median mask/DCS ratio, bias %, MAPE %, MAE (ha).
- **Known limits:** 500 m pixels still mix paddy with other land at field edges; DCS sown extent is itself an estimate; the mask is a single-sensor rule set with fixed thresholds.
- **Fix (2026-11-05):** the first run failed on every attempt with the deterministic Earth Engine error "Reprojection output too large (14025x7353 pixels)". Cause: the WorldCover 10 m layer was averaged to a 500 m grid (2,500 input pixels per output pixel) and the area sums were requested without an explicit scale. Changes:
  - cropland now uses the pilot's `cropland_fraction` on the 250 m grid (625 input pixels per output pixel, under the 1,024 limit);
  - every reduction passes `scale=250` with the MODIS crs and `tileScale=8`; no other image is reprojected;
  - `get_with_retry` now fails fast on deterministic errors (`DETERMINISTIC_EE_ERRORS`) and still retries timeouts and quota errors;
  - a test mode (`--districts LK72 --season Maha --seasons 2020-2020`) writes to `data/interim/ee/mask_test/` and every run checks that mask area ≤ cropland area.
- **Run result (2026-11-05), seasonal mask accepted as a detector, strict variant** (relaxed is nearly identical: flood-window clear share ≈ 1.00, so cloud is not the limit). From `results/tables/mask_validation.csv`, 52 seasons × 7 districts, 2000–2025:
  - **(a) It is a conservative core-paddy mask, not an area map.** Median mask area is 37% of DCS sown area (bias −64.5%, n = 364). Pooled Pearson r = 0.50 (Maha 0.38, Yala 0.60). The weakest agreement is Kurunegala (Maha r 0.46, Yala median ratio 0.06) and Batticaloa Maha (r 0.49, ratio 0.16); the best is Polonnaruwa Yala (r 0.90) and Trincomalee Yala (r 0.89). WorldCover cropland on the 250 m grid is itself smaller than the largest DCS sown area in several districts (e.g. Anuradhapura 94,881 ha vs Maha maximum 127,517 ha; Kurunegala 54,127 ha vs 82,478 ha), so the mask cannot reach DCS area there. Possible causes (not tested): paddy mapped to other WorldCover classes, the 50% cropland-fraction rule, mixed 500 m pixels.
  - **(b) Leakage: the seasonal mask must NOT be used for features.** The mask for season t uses season t's own flooding, green-up and harvest, all after the 12/8/4-week forecast cutoffs. It is kept only as a detector and for the area check above.

## Step 3 — leakage-free static mask and masked features (2026-11-05)
- **Static mask** (`src/data/static_mask.py`): a pixel is paddy for season type s if the strict seasonal rules detected it in at least k of the 9 same-season source seasons 2000–2008 (`features.paddy_mask.static.years`). All source seasons are strictly before the first forward-chaining test season, Maha 2009/10 (pilot fold 0, checked by a test). Separate masks for Maha and Yala.
- **Choosing k** (no yields): from the detection-frequency table (`data/interim/ee/mask/static_freq.csv`), k is the value whose pooled static-mask area over the 7 districts is closest to the pooled mean DCS sown extent of the same season in the **source years only** (2000–2008). Ties go to the larger k. The chosen k and the full k table are in `data/interim/ee/mask/static_k.json`. Can be fixed in config.
- **Assets:** the masks are exported as Earth Engine assets under `projects/paddy-yield-ee/assets/paddycast/` on the MODIS 250 m grid with mean pyramids. MODIS features use the 0/1 mask exactly; CHIRPS (5.5 km) and ERA5-Land (11 km) read it as the paddy-area fraction of each coarse pixel, so their district means are paddy-area weighted (the pilot used plain district means for these).
- **Validation:** static mask area vs DCS sown extent per district and season, mean ratio over 2000–2025 and separately for the source years and after → `results/tables/mask_static_validation.csv`.
- **Features** (`src/data/ee_extract_masked.py`, `src/data/build_masked_features.py`): the pilot's `features.py` is reused unchanged, so the cutoff, latency, past-only anomaly and label-publication rules are identical. Leads 12, 8, 4 weeks → `data/processed/masked/features_lead{L}.parquet`. Maha rows use the Maha mask, Yala rows the Yala mask. The unmasked pilot features for the same leads remain in `data/processed/variants/` for comparison.
- **Leakage tests** (`tests/test_static_mask.py`): source seasons are strictly before the first test season; the first test season matches the pilot's fold 0; source years past the first test season are rejected; the Earth Engine graph is built only from source seasons; the k rule ignores yields and post-source years.
- **Not done yet:** a rolling mask (seasons t−5 to t−1) is optional and not built.

## Step 4 — master tables and leakage tests (2026-11-05)
- **Tables** (`src/data/build_master_masked.py`): `data/processed/master_masked.parquet` (static-mask features) and `data/processed/master_unmasked_7d.parquet` (pilot WorldCover-cropland features), one row per district × season × lead_weeks (12, 8, 4) for the 7 study districts, every row with a DCS yield. Both tables have exactly the same rows, so Step 6 compares masked vs unmasked on identical folds and seeds. Columns `mask` and `lead_weeks` identify the variant; `mahaweli_affected` and `diag_post_merge_affected` are added as in the pilot.
- **Lagged yield features** (`y_prev_same`, `y_mean_same`, `y_last5_same`, `y_trend_same`, `y_last_avail`, `y_last_avail_age_days`, `sown_prev_same`, `harv_ratio_prev_same`, `n_label_hist_same`) come from the pilot `features.py`: previous seasons only, and only once that season's DCS figure counts as published at the forecast date (harvest reference + 6 months). They do not depend on the mask; the build checks they match the unmasked table (largest difference 1.5e-10 in `y_trend_same`, float noise, so the check uses atol 1e-6).
- **Forbidden columns** (`production_t`, `sown_ha`, `harvested_ha` of the target season) are checked absent.
- **Leakage tests on the real data** (`tests/test_leakage_masked.py`, skipped if the tables or masked series are missing):
  1. one row per district × season × lead; masked and unmasked rows identical; `forecast_date` = harvest reference − lead;
  2. **no data after the cutoff:** sampled rows rebuilt from series truncated at the forecast date (CHIRPS pentad end ≤ cutoff, MODIS composite end ≤ cutoff, ERA5 day < cutoff) equal the full-series rebuild and the stored row;
  3. **no future-season yield:** replacing every DCS label not yet published at the forecast date (asserted to include the target season and all later seasons) with random numbers changes no feature;
  4. MODIS anomalies equal value − mean of strictly earlier years of the same district, season and lead.
- **Negative control:** adding noise to stored `rain_b1` and `ndvi_b1_anom` makes tests 2 and 4 fail, so the tests can detect a change.

## Step 6 — training design, fixed before any full run (2026-11-05)
- **Code:** `src/experiments/masked_compare.py`, reusing the pilot's `run_model`, tuning grids, inner time splits, splitters, conformal quantile and SHAP code. Two private Kaggle kernels, CPU only, no internet: `06a_masked_models` (stage `models`) and `06b_masked_followups` (stage `followups`). Tables come from the saved predictions with the `report` stage, run locally, with no training.
- **Data:** private dataset `imalkaperera/paddycast-masked` (`master_masked.parquet`, `master_unmasked_7d.parquet`, `feature_list.json`). The 120 features are the pilot's four groups (lagged_yield, rainfall, modis, temp_radiation) plus `is_maha`.
- **Same folds, same seeds:** masked and unmasked tables have identical rows. Folds depend only on season, year and the label-publication rule, so both variants get exactly the same train/test indices at each lead (`tests/test_masked_compare.py`). Seeds: 42, 7, 123, 2024, 31337. Grids are those in `src/models/registry.py`.
- **Folds per lead:** forward chaining at leads 12, 8 and 4. Each lead has its own forecast date, so its own label-availability rule. 33 test seasons per lead, first Maha 2009/10, 231 test rows.
- **Models:** baseline_last5, ridge, random_forest, lightgbm, xgboost. The pilot's hybrid_lgbm is not run.
- **Leave-district-out:** first seed, fixed parameters chosen before running: ridge alpha 10 and LightGBM 300 / 0.03 / 7 leaves (pilot values); RF min_samples_leaf 15, max_features 0.5, and XGBoost 200 / 0.03 / depth 3 (the lowest-capacity points of their grids).
- **Conformal:** the pilot's split-conformal procedure at each lead. Calibration = the last 3 harvest years of each training window; models are fit on the rest. Standard, and Mondrian by season (Maha and Yala separately), for baseline, ridge and RF, seed 42, α = 0.1 (reliability α 0.5–0.05). With 7 districts a fold calibrates on about 21 Maha and 14 Yala residuals, so for Yala at α = 0.05 the interval is infinite. The report gives the share of infinite intervals instead of hiding them.
- **Ablation and SHAP:** RF, seed 42, pilot variants (all; lagged_yield only; lagged + one group; all minus one group; no lagged_yield), per mask × lead. SHAP is TreeExplainer on each fold's test rows only. Features with no observed training value at that lead (blocks after the cutoff) get SHAP 0.
- **Tests and Holm families:** paired Wilcoxon signed-rank on per-row absolute errors averaged over seeds, zeros dropped, effect = matched-pairs rank-biserial r. Holm correction within each family:
  1. masked vs unmasked, forward, 4 models × 3 leads (12 tests);
  2. model vs baseline, 2 masks × 3 leads × 4 models (24 tests);
  3. lead 12 and lead 8 vs lead 4, 2 masks × 5 models (20 tests);
  4. masked vs unmasked under leave-district-out (12 tests);
  5. ablation, within each mask × lead.
- **Evaluation rows:** every comparison uses only the rows predicted by every unit in it, and only where the baseline exists.
- **Smoke test only** (last 3 test seasons, sandbox): it checks the code path. Its numbers are not results and are not reported.

## Ridge mismatch in the pilot: RESULTS.md 620.2 vs results/predictions.csv 598.7 (2026-11-05)
- **Explanation:** the two numbers come from two different runs; there is no bug.
  - **620.2** is kernel 01 (`results/tables/model_comparison.csv`, `predictions_models.csv` seed 42): ridge tuned and fit on the **whole** training window of each fold.
  - **598.7** is kernel 02 (`results/tables/predictions_conformal.csv`, `conformal_point_metrics.csv`): ridge fit on the **proper** set only, i.e. the training window **minus its last 3 harvest years**, which are held back for conformal calibration.
- **Where it comes from:** `results/predictions.csv` is written by `src/analysis/report_figures.predictions_file()` as a copy of `predictions_conformal.csv`, so it carries the kernel-02 point predictions. Checked: the maximum absolute difference between `predictions.csv` and `predictions_conformal.csv` is 9.1e-13 kg/ha for every model.
- **Same rows:** both runs score the same 808 rows. The same file also explains the RF difference (605.8 in kernel 01 vs 607.0 in predictions.csv); the baseline is 605.8 in both.
- **What it means:** ridge does better *without* the 3 most recent training years (598.7 vs 620.2). This is one more sign that ridge is unstable here, consistent with its poor earliness results. It is not evidence that ridge is the best model. RESULTS.md must say which run each number comes from, and predictions.csv must be labelled as the conformal (proper-set) predictions.
