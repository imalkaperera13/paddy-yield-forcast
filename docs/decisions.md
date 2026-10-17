# Decisions log

Every modelling and data choice, with the reason. Newest entries at the bottom of each section.

## Task 0 — scaffold (2026-10-17)
- **Python target 3.11**, dependencies pinned in `requirements.txt`. The scaffold was built and tested on Python 3.13 in a Linux container; the real runs use 3.11 on Windows or Kaggle.
- **Config** lives only in `configs/config.yaml`. Code reads it through `src/config.py`.
- **Seeds**: base seed 42; a spread is reported over the five seeds in `project.seeds_for_spread`.
- **Tracking**: MLflow with a local file store (`./mlruns`, git-ignored).
- **Raw data** is never committed and never edited. Cleaning happens only in scripts.
- **No random k-fold** anywhere (`validation.random_kfold_allowed: false`, checked in tests).

## Task 1 — DCS yields (pending)
- Season-year convention (how Maha 2021/22 is labelled): _TBD_
- Mahaweli / non-Mahaweli handling: _TBD_
- Dropped district-seasons: _TBD_

## Task 2 — features (pending)
- Growth-stage windows: _TBD_
- Cropland mask: ESA WorldCover v200 class 40 (cropland). This is a proxy for paddy; WorldCover has no paddy class.
