"""Leakage-safe splitters. Every split is defined on `season_order` (harvest-time order) and
the label-availability rule, never on row position or random shuffling.

(a) forward_chaining      : test = all rows of ONE season; train = rows whose labels would have
                            been published by that season's forecast date (harvest_ref + lag
                            <= test forecast date). First test season chosen so the training
                            window already spans >= min_train_years distinct harvest years.
(b) leave_district_out    : the same folds, but the test district is removed from training
                            entirely (spatial generalisation; time order still respected).
(c) calibration_split     : inside a training window, the most recent `calibration_years`
                            harvest years become the calibration set (split conformal);
                            the rest is the proper training set.
Random k-fold is deliberately not provided; `random_kfold` raises.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.features import Calendar


@dataclass(frozen=True)
class Fold:
    fold_id: int
    test_season: str
    test_season_year: int
    test_order: int
    forecast_date: pd.Timestamp
    train_idx: np.ndarray
    test_idx: np.ndarray
    held_out_district: str | None = None


def label_published(df: pd.DataFrame, cal: Calendar) -> pd.Series:
    """Date each row's label becomes public: its reference harvest + assumed lag."""
    lag = pd.DateOffset(months=cal.lag_months)
    return pd.Series([cal.harvest_ref(s, y) + lag
                      for s, y in zip(df.season, df.season_year, strict=True)], index=df.index)


def _test_seasons(df: pd.DataFrame, cal: Calendar, min_train_years: int, lead_weeks: int):
    seasons = (df[["season", "season_year", "season_order", "harvest_year"]]
               .drop_duplicates().sort_values("season_order"))
    pub = label_published(df, cal)
    out = []
    for r in seasons.itertuples(index=False):
        fdate = cal.forecast_date(r.season, r.season_year, lead_weeks)
        train = df[pub <= fdate]
        if train.harvest_year.nunique() >= min_train_years:
            out.append((r.season, r.season_year, r.season_order, fdate))
    return out, pub


def forward_chaining(df: pd.DataFrame, cal: Calendar, min_train_years: int = 10,
                     lead_weeks: int = 0, seasons: tuple[str, ...] = ("Maha", "Yala")
                     ) -> Iterator[Fold]:
    tests, pub = _test_seasons(df, cal, min_train_years, lead_weeks)
    k = 0
    for season, sy, order, fdate in tests:
        if season not in seasons:
            continue
        test_idx = df.index[(df.season == season) & (df.season_year == sy)].to_numpy()
        train_idx = df.index[(pub <= fdate) & (df.season_order < order)].to_numpy()
        yield Fold(k, season, sy, order, fdate, train_idx, test_idx)
        k += 1


def leave_district_out(df: pd.DataFrame, cal: Calendar, min_train_years: int = 10,
                       lead_weeks: int = 0, seasons: tuple[str, ...] = ("Maha", "Yala")
                       ) -> Iterator[Fold]:
    k = 0
    for f in forward_chaining(df, cal, min_train_years, lead_weeks, seasons):
        for d in sorted(df.loc[f.test_idx, "district"].unique()):
            test_idx = f.test_idx[df.loc[f.test_idx, "district"].to_numpy() == d]
            train_idx = f.train_idx[df.loc[f.train_idx, "district"].to_numpy() != d]
            yield Fold(k, f.test_season, f.test_season_year, f.test_order, f.forecast_date,
                       train_idx, test_idx, held_out_district=d)
            k += 1


def calibration_split(df: pd.DataFrame, train_idx: np.ndarray, calibration_years: int = 3
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Most recent `calibration_years` harvest years inside the training window -> calibration."""
    hy = df.loc[train_idx, "harvest_year"]
    cut = sorted(hy.unique())[-calibration_years]
    proper = train_idx[(hy < cut).to_numpy()]
    calib = train_idx[(hy >= cut).to_numpy()]
    return proper, calib


def inner_time_splits(df: pd.DataFrame, train_idx: np.ndarray, n_splits: int = 3
                      ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Time-aware inner splits for hyper-parameter tuning: expanding window over the last
    `n_splits` harvest years of the training set (each validated on one later year)."""
    hy = df.loc[train_idx, "harvest_year"].to_numpy()
    years = sorted(set(hy))
    out = []
    for y in years[-n_splits:]:
        out.append((train_idx[hy < y], train_idx[hy == y]))
    return out


def random_kfold(*_args, **_kwargs):
    raise RuntimeError("Random k-fold is not allowed for main results (leakage across time). "
                       "Use src.analysis demo code for the documented comparison only.")


def fold_table(folds: list[Fold], df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in folds:
        tr = df.loc[f.train_idx]
        rows.append({"fold": f.fold_id, "test_season": f.test_season,
                     "test_season_year": f.test_season_year,
                     "forecast_date": f.forecast_date.date(), "n_train": len(f.train_idx),
                     "n_test": len(f.test_idx), "train_first_harvest": tr.harvest_year.min(),
                     "train_last_harvest": tr.harvest_year.max(),
                     "train_years": tr.harvest_year.nunique(),
                     "held_out_district": f.held_out_district})
    return pd.DataFrame(rows)
