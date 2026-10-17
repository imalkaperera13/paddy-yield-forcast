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
