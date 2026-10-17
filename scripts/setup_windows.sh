#!/usr/bin/env bash
# Step A setup for Windows + Git Bash. Run from the repo root:
#     bash scripts/setup_windows.sh
# Everything is written to setup_log.txt (git-ignored) so Claude can read it.
# It never prints or reads your Kaggle / Earth Engine credentials.

LOG=setup_log.txt
exec > >(tee "$LOG") 2>&1
step() { echo; echo "=== $* ==="; }

step "1. System"
date
uname -a
echo "shell: $SHELL ($BASH_VERSION)"
git --version
git status -sb | head -3
git remote -v | sed -E 's#https://[^@]*@#https://#'   # hide any token in the URL

step "2. Python 3.11"
if py -3.11 --version 2>/dev/null; then
  PY="py -3.11"
else
  echo "Python 3.11 NOT FOUND (py -3.11 failed). Installed versions:"
  py -0p 2>/dev/null || python --version
  echo "STOP: install Python 3.11 first (see Claude's instructions), then re-run."
  exit 1
fi

step "3. Virtual environment .venv"
if [ ! -d .venv ]; then $PY -m venv .venv; fi
source .venv/Scripts/activate
python --version
which python

step "4. Install requirements (this takes several minutes)"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
PIP_STATUS=$?
echo "pip install exit code: $PIP_STATUS"
python -m pip freeze > requirements-freeze.txt
echo "frozen $(wc -l < requirements-freeze.txt) packages to requirements-freeze.txt"

step "5. Import check (geo stack is the usual Windows trouble-maker)"
python - <<'EOF'
import importlib
for m in ["numpy","pandas","scipy","sklearn","lightgbm","xgboost","shap","statsmodels",
          "pyarrow","yaml","ee","geemap","geopandas","rasterio","shapely","mlflow",
          "matplotlib","seaborn","pytest","kaggle"]:
    try:
        mod = importlib.import_module(m)
        print(f"  ok   {m:12s} {getattr(mod,'__version__','')}")
    except Exception as e:
        print(f"  FAIL {m:12s} {type(e).__name__}: {str(e)[:120]}")
EOF

step "6. Project checks"
python scripts/env_check.py
python -m pytest
python -m ruff check src tests scripts kaggle

step "7. Kaggle CLI (token is NOT printed)"
if [ -f "$HOME/.kaggle/kaggle.json" ] || [ -f "$HOME/.kaggle/access_token" ] || [ -n "$KAGGLE_API_TOKEN" ]; then
  echo "token file/env found"
  kaggle --version
  kaggle datasets list -s test 2>&1 | head -4
else
  echo "no Kaggle token yet (expected until you do the Kaggle step)"
fi

step "8. Earth Engine"
if earthengine --help >/dev/null 2>&1; then echo "earthengine CLI installed"; fi
if [ -f "$HOME/.config/earthengine/credentials" ]; then
  echo "EE credentials file present (not printed)"
else
  echo "not authenticated yet (expected until you do the Earth Engine step)"
fi

step "DONE"
