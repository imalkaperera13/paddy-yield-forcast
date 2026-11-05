"""Step 4: master tables for the 7 study districts, masked and unmasked, leads 12/8/4.

    .venv/Scripts/python -m src.data.build_master_masked

Inputs
- data/processed/masked/features_lead{12,8,4}.parquet   static-mask features (Step 3)
- data/processed/variants/master_lead{12,8,4}_shift0.parquet   pilot unmasked features
Outputs
- data/processed/master_masked.parquet    masked features, one row per
                                           district x season x lead_weeks
- data/processed/master_unmasked_7d.parquet   pilot unmasked features for exactly the same
                                           rows, so Step 6 compares the two on identical folds

Every row is a district-season with a DCS yield (2000/01 onward). Lagged yield features
(y_prev_same, y_mean_same, y_last5_same, y_trend_same, y_last_avail, sown_prev_same,
harv_ratio_prev_same) come from previous seasons only and only once that season's DCS figure
counts as published at the forecast date (6-month lag assumption). They are identical in the
two tables because they do not depend on the mask.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.config import REPO_ROOT, load_config

PROC = REPO_ROOT / "data" / "processed"
LEADS = (12, 8, 4)
KEY = ["district", "season", "season_year", "lead_weeks"]
OUT_MASKED = PROC / "master_masked.parquet"
OUT_UNMASKED = PROC / "master_unmasked_7d.parquet"


def main() -> None:
    cfg = load_config()
    sel = pd.read_csv(REPO_ROOT / cfg["data"]["selected_districts_file"])
    districts = set(sel["district"])
    affected = set(cfg["data"]["mahaweli_affected_districts"])
    forbidden = set(cfg["forbidden_features"])

    masked, unmasked = [], []
    for lead in LEADS:
        m = pd.read_parquet(PROC / "masked" / f"features_lead{lead}.parquet")
        u = pd.read_parquet(PROC / "variants" / f"master_lead{lead}_shift0.parquet")
        u = u[u.district.isin(districts)]
        m["mahaweli_affected"] = m.district.isin(affected).astype(int)
        m["diag_post_merge_affected"] = m.mahaweli_affected * m.mahaweli_merged
        rows = m[["district", "season", "season_year"]]
        u = rows.merge(u, on=["district", "season", "season_year"], how="left",
                       validate="one_to_one")
        if u["yield_kg_ha"].isna().any():
            raise RuntimeError(f"lead {lead}: unmasked table is missing rows")
        u["mask"] = "worldcover_cropland"
        m["lead_weeks"], u["lead_weeks"] = lead, lead
        masked.append(m)
        unmasked.append(u[m.columns])

    M = pd.concat(masked, ignore_index=True).sort_values(KEY).reset_index(drop=True)
    U = pd.concat(unmasked, ignore_index=True).sort_values(KEY).reset_index(drop=True)
    for name, df in (("masked", M), ("unmasked", U)):
        if df.duplicated(KEY).any():
            raise RuntimeError(f"{name}: duplicate district x season x lead rows")
        bad = forbidden & set(df.columns) - {"yield_kg_ha", "mahaweli_merged", "season_year",
                                              "harvest_year", "season_order"}
        if bad:
            raise RuntimeError(f"{name}: forbidden feature columns {bad}")
    if not (M[KEY].values == U[KEY].values).all():
        raise RuntimeError("masked and unmasked tables do not have identical rows")
    lag_cols = [c for c in M.columns if c.startswith("y_") or c in
                ("sown_prev_same", "harv_ratio_prev_same", "n_label_hist_same")]
    # tolerance only absorbs float noise (y_trend_same differs by ~1e-10 from the pilot run)
    same = np.allclose(M[lag_cols].to_numpy(float), U[lag_cols].to_numpy(float),
                       rtol=0, atol=1e-6, equal_nan=True)
    if not same:
        raise RuntimeError("lagged yield features differ between masked and unmasked tables")

    M.to_parquet(OUT_MASKED, index=False)
    U.to_parquet(OUT_UNMASKED, index=False)
    print(f"wrote {OUT_MASKED.relative_to(REPO_ROOT)}: {M.shape[0]} rows x {M.shape[1]} cols")
    print(f"wrote {OUT_UNMASKED.relative_to(REPO_ROOT)}: {U.shape[0]} rows x {U.shape[1]} cols")
    print(M.groupby(["lead_weeks", "season"]).size().rename("rows").to_string())


if __name__ == "__main__":
    main()
