"""Task 0 sanity tests: config is valid and the no-leakage guard rails are set."""

from pathlib import Path

from src.config import REPO_ROOT, load_config


def test_config_loads():
    cfg = load_config()
    assert cfg["project"]["seed"] == 42
    assert cfg["data"]["seasons"] == ["Maha", "Yala"]


def test_no_random_kfold_and_no_imputation():
    cfg = load_config()
    assert cfg["validation"]["random_kfold_allowed"] is False
    assert cfg["data"]["impute_missing"] is False


def test_raw_data_is_gitignored():
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("data/raw/*", ".env", "mlruns/"):
        assert pattern in text


def test_repo_structure():
    for d in ["configs", "data/raw", "data/interim", "data/processed", "src/data",
              "src/validation", "src/models", "src/analysis", "tests", "notebooks",
              "results/tables", "results/figures", "docs"]:
        assert (REPO_ROOT / d).is_dir(), d
    assert Path(REPO_ROOT / "docs" / "decisions.md").is_file()
