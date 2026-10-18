"""Model-level leakage and sanity tests on SYNTHETIC data."""

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.dcs import season_fields
from src.models.metrics import metrics
from src.models.train import run_model

CFG = load_config()


def synthetic_master(seed=0):
    """SYNTHETIC master-like table: 5 districts x 2 seasons x 2000-2020."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in "ABCDE":
        for y in range(2000, 2021):
            for s, lab in (("Maha", f"{y}/{y + 1}"), ("Yala", y)):
                rows.append({"district": d, "season": s, **season_fields(s, lab)})
    df = pd.DataFrame(rows)
    df["f1"], df["f2"] = rng.normal(size=len(df)), rng.normal(size=len(df))
    df["yield_kg_ha"] = 3500 + 400 * df.f1 + rng.normal(0, 100, len(df))
    df["y_last5_same"] = 3500.0
    for c in ("mahaweli_merged", "mahaweli_affected", "diag_post_merge_affected", "sanity_flag"):
        df[c] = 0
    return df


def test_SYNTHETIC_future_targets_do_not_change_past_predictions(tmp_path):
    df = synthetic_master()
    p0 = run_model(df, ["f1", "f2"], "ridge", 42, CFG, tmp_path / "a",
                   fixed_params={"alpha": 1.0}, log=lambda *_: None)
    cut = int(p0.season_order.median())
    df2 = df.copy()
    df2.loc[df2.season_order >= cut, "yield_kg_ha"] = 1e6      # poison the future
    p1 = run_model(df2, ["f1", "f2"], "ridge", 42, CFG, tmp_path / "b",
                   fixed_params={"alpha": 1.0}, log=lambda *_: None)
    early0 = p0[p0.season_order < cut].sort_values(["district", "season_order"])
    early1 = p1[p1.season_order < cut].sort_values(["district", "season_order"])
    np.testing.assert_allclose(early0.y_pred.to_numpy(), early1.y_pred.to_numpy())


def test_SYNTHETIC_model_beats_flat_baseline_when_signal_exists(tmp_path):
    df = synthetic_master()
    pr = run_model(df, ["f1", "f2"], "ridge", 42, CFG, tmp_path, log=lambda *_: None)
    pb = run_model(df, ["f1", "f2"], "baseline_last5", 42, CFG, tmp_path, log=lambda *_: None)
    assert metrics(pr.y_true, pr.y_pred)["rmse"] < metrics(pb.y_true, pb.y_pred)["rmse"]


def test_SYNTHETIC_results_are_cached_for_resume(tmp_path):
    df = synthetic_master()
    run_model(df, ["f1"], "baseline_last5", 1, CFG, tmp_path, log=lambda *_: None)
    msgs = []
    run_model(df, ["f1"], "baseline_last5", 1, CFG, tmp_path, log=msgs.append)
    assert any("cached" in m for m in msgs)
