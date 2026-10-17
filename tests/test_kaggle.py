"""Guard rails for the Kaggle workflow: private, CPU-only, offline, no tokens in the repo."""

import json

from src.config import REPO_ROOT, load_config

KERNELS = sorted((REPO_ROOT / "kaggle" / "kernels").glob("*/kernel-metadata.json"))


def test_five_experiment_kernels_exist():
    assert len(KERNELS) == 5


def test_kernels_private_cpu_offline():
    for p in KERNELS:
        meta = json.loads(p.read_text(encoding="utf-8"))
        assert meta["is_private"] is True, p
        assert meta["enable_gpu"] is False, p
        assert meta["enable_internet"] is False, p
        assert meta["code_file"] == "run.py" and (p.parent / "run.py").is_file(), p
        assert len(meta["dataset_sources"]) == 2, p


def test_kernel_slugs_match_config():
    cfg = load_config()["kaggle"]
    for p in KERNELS:
        run = (p.parent / "run.py").read_text(encoding="utf-8")
        assert f'CODE_SLUG = "{cfg["code_slug"]}"' in run
        assert f'DATA_SLUG = "{cfg["data_slug"]}"' in run


def test_no_kaggle_token_in_repo():
    for name in ("kaggle.json", "access_token"):
        assert not list(REPO_ROOT.rglob(name)), f"{name} must never be inside the repo"
    gi = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "kaggle.json" in gi and "access_token" in gi


def test_output_dir_is_local_when_not_on_kaggle(tmp_path, monkeypatch):
    from src import kaggle_env
    assert kaggle_env.ON_KAGGLE is False
    out = kaggle_env.output_dir("unit_test_tmp")
    assert out.is_dir() and "results" in out.parts
    out.rmdir()
