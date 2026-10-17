# Results explained

> **Status: no results yet.** Every table and figure below is filled in only from code that has been run. Each one cites its Kaggle kernel version and git commit.

For each item, the explanation covers:
- what it shows,
- how to read it,
- what it does **not** prove,
- why it may have happened. These reasons are hypotheses unless they were tested.

## Model comparison
_pending (Task 5)_

## Paired Wilcoxon vs baseline
_pending (Task 6)_

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
_pending: this needs a demonstration run that scores the same model with random k-fold and with forward-chaining (kernel 01_models)._

## Limitations observed
_Filled from the data as work proceeds. Items known so far (Task 1, from the DCS files):_
- **Label noise:** DCS yield is a crop-cutting sample estimate, so the label itself is noisy.
- **Mahaweli H / Udawalawe merge:** until 2017/18 Maha and 2018 Yala these irrigation areas were reported separately; from then on they are inside their host districts. This structural break could not be measured cleanly (the 2016/17 drought falls just before it). Results will be reported (a) for all districts, (b) excluding post-merge rows of the affected districts, and (c) as error before vs after the merge. `mahaweli_merged` is not used as a model feature.
- **Missing district-seasons:** 101 of 2,375 are dropped. Jaffna has no Yala yield after 1983, and northern and eastern districts have gaps in the conflict years.
- **Raw-file defects:** the Production-Maha row order and two header typos were repaired in code. Because the production repair relied on yield × harvested area, production is not an independent check on yield.
- **Label availability:** the 6-month publication lag is an assumption, not a verified DCS release schedule.
- **Short feature history:** satellite features start in 2000, so the modelling table has about 25 seasons per district, not 47.
- **Paddy mask proxy (measured):** WorldCover 2021 class 40 at 250 m (≥50% cropland) covers 0.85× the DCS 2018–2024 mean Maha sown area nationally. In the dry zone it is 0.7–1.3×, but in the wet zone it covers only ~0.10–0.47× (Kegalle 0.10, Galle 0.14, Kalutara 0.18, Nuwara Eliya 0.22, Colombo 0.27, Kandy 0.37, Matara/Ratnapura 0.40, Gampaha 0.47). There, wet-zone paddy is often mapped as grassland or wetland, so NDVI rests on fewer pixels. Possible sensitivity test: add class 90 (herbaceous wetland) or lower the 50% threshold.
