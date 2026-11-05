"""Step 6 checks that need no model training."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.config import REPO_ROOT, load_config
from src.data.features import Calendar
from src.models.train import folds_for
from src.validation.splitters import calibration_split, label_published

from src.experiments import masked_compare as mc

PROC = REPO_ROOT / "data" / "processed"
HAVE = (PROC / mc.MASKS["masked"]).exists() and (PROC / mc.MASKS["unmasked"]).exists()


def test_holm():
    assert np.allclose(mc.holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])


def test_paired_test_sign():
    r = mc.paired_test(pd.Series([5.0, 3.0, 2.0, 4.0, 1.0, 0.0]))
    assert r["share_better"] == 1.0 and r["rank_biserial_r"] == 1.0 and r["n_nonzero"] == 5


@pytest.mark.parametrize("tag,kind,mask,lead,variant", [
    ("masked_L12", None, "masked", "12", None),
    ("ldo_unmasked_L4", "ldo_", "unmasked", "4", None),
    ("conformal_masked_L8", "conformal_", "masked", "8", None),
    ("abl_unmasked_L8_all_minus_modis", "abl_", "unmasked", "8", "all_minus_modis"),
])
def test_tag_parsing(tag, kind, mask, lead, variant):
    m = mc.TAG_RE.match(tag)
    assert (m["kind"], m["mask"], m["lead"], m["variant"]) == (kind, mask, lead, variant)


@pytest.fixture(scope="module")
def tabs():
    if not HAVE:
        pytest.skip("master tables not built")
    return mc.load_tables(PROC, load_config())[0]


@pytest.mark.parametrize("lead", mc.LEADS)
def test_same_folds_for_masked_and_unmasked(tabs, lead):
    cfg = load_config()
    cal = Calendar.from_cfg(cfg)
    a = folds_for(mc.lead_table(tabs, "masked", lead), cal, cfg, "forward", lead)
    b = folds_for(mc.lead_table(tabs, "unmasked", lead), cal, cfg, "forward", lead)
    assert len(a) == len(b) > 0
    for fa, fb in zip(a, b, strict=True):
        assert (fa.test_season, fa.test_season_year) == (fb.test_season, fb.test_season_year)
        assert np.array_equal(fa.train_idx, fb.train_idx)
        assert np.array_equal(fa.test_idx, fb.test_idx)


@pytest.mark.parametrize("lead", mc.LEADS)
def test_training_and_calibration_labels_published_before_cutoff(tabs, lead):
    cfg = load_config()
    cal = Calendar.from_cfg(cfg)
    df = mc.lead_table(tabs, "masked", lead)
    pub = label_published(df, cal)
    for f in folds_for(df, cal, cfg, "forward", lead):
        assert (pub.loc[f.train_idx] <= f.forecast_date).all()
        assert (df.loc[f.train_idx, "season_order"] < f.test_order).all()
        proper, calib = calibration_split(df, f.train_idx, cfg["validation"]["calibration_years"])
        assert set(calib) <= set(f.train_idx) and not set(calib) & set(proper)
        assert df.loc[proper, "harvest_year"].max() < df.loc[calib, "harvest_year"].min()


def test_ldo_params_fixed_for_every_model():
    assert set(mc.LDO_FIXED) == set(mc.MODEL_ORDER)
