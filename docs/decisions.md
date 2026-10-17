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

## Task 1 — DCS yields (pending)
- Season-year convention (how Maha 2021/22 is labelled): _TBD_
- Mahaweli / non-Mahaweli handling: _TBD_
- Dropped district-seasons: _TBD_

## Task 2 — features (pending)
- Growth-stage windows: _TBD_
- Cropland mask: ESA WorldCover v200 class 40 (cropland). This is a proxy for paddy; WorldCover has no paddy class.
