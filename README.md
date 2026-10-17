# Paddy yield forecast (Stage 1: ML)

Calibrated early-warning forecasting of district-level paddy (rice) yield in Sri Lanka. The labels are official DCS yields; the features come from satellite and climate data.
Final-year project, BSc Computer Science, University of Bedfordshire (CIS013-3 / CIS017-3).

## Quick start (Windows, Git Bash)

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
python scripts/env_check.py   # environment report
python -m pytest              # tests
python -m src.run_all         # full pipeline (stages are added task by task)
```

On Kaggle, run `pip install -r requirements.txt` in the notebook and upload the processed master table as a **private** dataset.

## Layout

```
configs/config.yaml        all settings (windows, seeds, splits, alpha)
data/{raw,interim,processed}   git-ignored; raw files are never edited
src/data                   DCS cleaning, Earth Engine extraction, master table
src/validation             leakage-safe splitters (forward-chaining, leave-district-out)
src/models                 baselines, ridge, RF, LightGBM, XGBoost, hybrid
src/analysis               statistics, conformal, earliness, ablation, SHAP, shock tests
tests/                     pytest, including the leakage tests
results/{tables,figures}   real outputs only, figures at 300 dpi
docs/                      decisions.md, data_dictionary.md
```

## Rules

- Every reported number comes from code that was actually run, and is traceable to a run ID and commit.
- No random k-fold. Anything fitted on the data uses only seasons before the one being predicted.
- Synthetic data is used only to test code, and is always labelled `SYNTHETIC`.

## Status

| task | status |
|---|---|
| 0 environment and scaffold | done |
| 1 DCS yield table | waiting for raw data |
| 2–11 | not started |
