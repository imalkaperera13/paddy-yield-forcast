"""Splitter leakage tests on a SYNTHETIC grid and, if present, on the real master table."""

import pandas as pd
import pytest

from src.config import REPO_ROOT, load_config
from src.data.dcs import season_fields
from src.data.features import Calendar
from src.validation.splitters import (
    calibration_split,
    forward_chaining,
    inner_time_splits,
    label_published,
    leave_district_out,
    random_kfold,
)

CFG = load_config()
CAL = Calendar.from_cfg(CFG)
MIN_YEARS = CFG["validation"]["min_train_years"]
MASTER = REPO_ROOT / "data" / "processed" / "master_table.parquet"


def synthetic_grid():
    """SYNTHETIC: 4 districts x 2 seasons x 2000-2024, no data values needed."""
    rows = []
    for d in "ABCD":
        for y in range(2000, 2025):
            rows.append({"district": d, "season": "Maha", **season_fields("Maha", f"{y}/{y + 1}")})
            rows.append({"district": d, "season": "Yala", **season_fields("Yala", y)})
    return pd.DataFrame(rows)


DATASETS = [pytest.param(synthetic_grid(), id="SYNTHETIC")]
if MASTER.exists():
    DATASETS.append(pytest.param(pd.read_parquet(MASTER), id="real_master"))


@pytest.mark.parametrize("df", DATASETS)
@pytest.mark.parametrize("lead", [0, 6, 12])
def test_forward_chaining_never_trains_on_future_or_unpublished(df, lead):
    pub = label_published(df, CAL)
    folds = list(forward_chaining(df, CAL, MIN_YEARS, lead))
    assert folds, "no folds produced"
    for f in folds:
        tr, te = df.loc[f.train_idx], df.loc[f.test_idx]
        assert (te.season_order == f.test_order).all()
        assert (tr.season_order < f.test_order).all(), "future/same season in training"
        assert (pub.loc[f.train_idx] <= f.forecast_date).all(), "label not yet published"
        assert not set(f.train_idx) & set(f.test_idx)
        assert tr.harvest_year.nunique() >= MIN_YEARS


@pytest.mark.parametrize("df", DATASETS)
def test_first_test_season_has_min_training_years(df):
    f0 = next(forward_chaining(df, CAL, MIN_YEARS))
    assert df.loc[f0.train_idx].harvest_year.nunique() >= MIN_YEARS


@pytest.mark.parametrize("df", DATASETS)
def test_leave_district_out_removes_district_and_respects_time(df):
    for f in list(leave_district_out(df, CAL, MIN_YEARS))[:200]:
        tr, te = df.loc[f.train_idx], df.loc[f.test_idx]
        assert (te.district == f.held_out_district).all()
        assert f.held_out_district not in set(tr.district)
        assert (tr.season_order < f.test_order).all()


@pytest.mark.parametrize("df", DATASETS)
def test_calibration_is_latest_years_inside_training(df):
    k = CFG["validation"]["calibration_years"]
    for f in forward_chaining(df, CAL, MIN_YEARS):
        proper, calib = calibration_split(df, f.train_idx, k)
        hp, hc = df.loc[proper, "harvest_year"], df.loc[calib, "harvest_year"]
        assert set(proper) | set(calib) == set(f.train_idx) and not set(proper) & set(calib)
        assert hc.nunique() == k and hp.max() < hc.min()
        assert (df.loc[calib, "season_order"] < f.test_order).all()


@pytest.mark.parametrize("df", DATASETS)
def test_inner_tuning_splits_are_time_ordered(df):
    f = next(forward_chaining(df, CAL, MIN_YEARS))
    for tr, va in inner_time_splits(df, f.train_idx, 3):
        assert df.loc[tr, "harvest_year"].max() < df.loc[va, "harvest_year"].min()
        assert set(va) <= set(f.train_idx)


def test_SYNTHETIC_injected_future_row_is_caught():
    """A deliberately broken split must fail the same check the tests use."""
    df = synthetic_grid()
    f = next(forward_chaining(df, CAL, MIN_YEARS))
    future = df.index[df.season_order > f.test_order][:1]
    bad_train = df.loc[list(f.train_idx) + list(future)]
    assert not (bad_train.season_order < f.test_order).all()


def test_random_kfold_is_blocked():
    with pytest.raises(RuntimeError):
        random_kfold()
