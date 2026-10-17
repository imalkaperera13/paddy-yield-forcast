# Training on Kaggle

Data is prepared locally. Training and evaluation run on Kaggle. Results are pulled back into `results/`.

## Kaggle limits we design around

Checked 2026-10-17. Confirm them in your own Kaggle settings, because they change.

| limit | value | how this project handles it |
|---|---|---|
| max session length | 12 h (CPU and GPU), 9 h (TPU) | each experiment group is a separate kernel; work is saved per fold or seed so a stopped run resumes |
| weekly GPU quota | ~30 h (may vary) | not used: tree models run on CPU (`enable_gpu: false`) |
| `/kaggle/working` | ~20 GB, auto-saved | outputs are CSV and PNG, well under 1 GB |
| RAM | ~30 GB CPU | the master table is about 1,000–2,000 rows, so memory is not a concern |
| phone verification | needed to unlock GPU/TPU (and, by Kaggle convention, internet) | kernels need neither: code arrives as a private dataset |
| dataset visibility | **private by default**; public only with `-u/--public` | never pass `--public` |

Warning: about 1,000 district-season rows is small for a neural network. An LSTM can easily overfit, so it runs only if you ask for one.

## One-time setup (you do this, never paste the token in chat)

1. Easiest (Kaggle CLI 2.x): run `.venv/Scripts/kaggle auth login`, which opens the browser; log in and allow. No token file to manage.
   Alternative: kaggle.com → profile picture → Settings → API → **Generate New Token**, and save the string as `~/.kaggle/access_token`. Then run `chmod 600 ~/.kaggle/access_token`.
2. The `kaggle` CLI is already installed by `requirements.txt` (pinned 2.2.4).
3. Set `kaggle.username` in `configs/config.yaml` (your username is not a secret).

## Each round

```bash
git commit ...                                   # manifest records the commit hash
python scripts/kaggle_prepare.py data            # master table + manifest.json + metadata
python scripts/kaggle_prepare.py code            # src/ + configs/ snapshot + COMMIT file
python scripts/kaggle_prepare.py kernels         # fills your username into kernel metadata

# first time
kaggle datasets create -p kaggle/_build/data -t          # -t keeps parquet as-is (no CSV conversion)
kaggle datasets create -p kaggle/_build/code -r zip
# afterwards
kaggle datasets version -p kaggle/_build/data -t -m "master table <commit>"
kaggle datasets version -p kaggle/_build/code -r zip -m "code <commit>"

kaggle kernels push -p kaggle/_build/kernels/01_models
kaggle kernels status <username>/paddy-01-baselines-and-models
kaggle kernels output <username>/paddy-01-baselines-and-models -p results/runs/01_models
```

Record each dataset version number and kernel version in `docs/decisions.md`. Every number in the report then points back to an exact run.

Check `kaggle kernels push --help` before the first push, because the CLI changes.

## Why the code is a private dataset and not a git clone

Cloning a private GitHub repo needs internet plus a token stored in a Kaggle Secret. A code dataset needs neither. It is versioned, and the `COMMIT` file inside it ties every run to an exact commit.

## Library versions on Kaggle

With internet off, kernels use Kaggle's preinstalled library versions, which may differ from `requirements.txt`. Each kernel prints the exact versions at the top and saves them to `run_info.json`, and those are the versions the report quotes.
