"""Model definitions and small, time-aware tuning grids.

Grids are deliberately small: ~1,000 rows and ~25 seasons cannot support heavy tuning, and
every extra configuration is another chance to overfit the inner validation years.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASELINE_COL = "y_last5_same"     # mean of the last 5 published same-season yields


@dataclass
class ModelSpec:
    name: str
    grid: list[dict] = field(default_factory=lambda: [{}])
    residual: bool = False        # True -> learn y - baseline, predict baseline + f(x)
    seeded: bool = True

    def build(self, params: dict, seed: int):
        if self.name == "ridge":
            return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 Ridge(alpha=params["alpha"]))
        if self.name == "random_forest":
            return make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(
                n_estimators=200, min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"], random_state=seed, n_jobs=-1))
        if self.name in ("lightgbm", "hybrid_lgbm"):
            from lightgbm import LGBMRegressor
            return LGBMRegressor(n_estimators=params["n_estimators"],
                                 learning_rate=params["learning_rate"],
                                 num_leaves=params["num_leaves"], min_child_samples=20,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                                 reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1)
        if self.name == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=params["n_estimators"],
                                learning_rate=params["learning_rate"],
                                max_depth=params["max_depth"], subsample=0.8,
                                colsample_bytree=0.7, reg_lambda=1.0, min_child_weight=5,
                                random_state=seed, n_jobs=-1, tree_method="hist")
        raise ValueError(self.name)


def _grid(**kw) -> list[dict]:
    keys = list(kw)
    out = [{}]
    for k in keys:
        out = [{**o, k: v} for o in out for v in kw[k]]
    return out


MODELS: dict[str, ModelSpec] = {
    "ridge": ModelSpec("ridge", _grid(alpha=list(np.logspace(-1, 3, 5))), seeded=False),
    "random_forest": ModelSpec("random_forest",
                               _grid(min_samples_leaf=[5, 15], max_features=[0.5])),
    "lightgbm": ModelSpec("lightgbm", _grid(n_estimators=[200, 500], learning_rate=[0.03],
                                            num_leaves=[7, 15])),
    "xgboost": ModelSpec("xgboost", _grid(n_estimators=[200, 500], learning_rate=[0.03],
                                          max_depth=[3, 5])),
    "hybrid_lgbm": ModelSpec("hybrid_lgbm", _grid(n_estimators=[100, 300], learning_rate=[0.03],
                                                  num_leaves=[7]), residual=True),
}
