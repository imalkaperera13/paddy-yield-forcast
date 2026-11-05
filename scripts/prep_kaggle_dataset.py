"""Step 5: fill kaggle_dataset/ for a PRIVATE Kaggle dataset. Uploads nothing.

    .venv/Scripts/python scripts/prep_kaggle_dataset.py <kaggle_username>
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1]
OUT = R / "kaggle_dataset"
FILES = {
    "master_masked.parquet": "data/processed/master_masked.parquet",
    "master_unmasked_7d.parquet": "data/processed/master_unmasked_7d.parquet",
    "selected_districts.csv": "data/processed/selected_districts.csv",
    "mask_validation.csv": "results/tables/mask_validation.csv",
    "mask_static_validation.csv": "results/tables/mask_static_validation.csv",
}
META = {
    "district": "district name (DCS)", "adm2_pcode": "district code",
    "season": "Maha or Yala", "season_label": "DCS season label",
    "season_year": "year the season starts", "harvest_year": "year of harvest",
    "season_order": "2*harvest_year + (0 Maha, 1 Yala); used for time ordering",
    "yield_kg_ha": "TARGET: DCS yield, kg/ha", "mahaweli_merged": "DCS merged Mahaweli H into district",
    "mahaweli_affected": "district affected by the Mahaweli merge",
    "diag_post_merge_affected": "affected x merged", "sanity_flag": "DCS row sanity flag",
    "forecast_date": "cutoff: harvest reference - lead", "lead_weeks": "12, 8 or 4",
    "shift_weeks": "always 0", "mask": "static_2000_2008 or worldcover_cropland",
}


def describe(c: str) -> str:
    if c in META:
        return META[c]
    if c.endswith("_anom"):
        return "value minus mean of earlier years only (same district, season, lead)"
    if c.startswith(("y_", "sown_prev", "harv_ratio_prev", "n_label_hist")):
        return "past seasons only, once published (6-month lag)"
    if c[:4] in ("ndvi", "evi_"):
        return "MODIS MOD13Q1 block feature, composites ending on/before cutoff"
    if c.startswith("rain"):
        return "CHIRPS block feature, pentads ending on/before cutoff"
    if c.startswith(("tmean", "tmax", "tmin", "heat", "era5")):
        return "ERA5-Land block feature, days before cutoff"
    return "see src/data/features.py"


def main() -> None:
    user = sys.argv[1]
    OUT.mkdir(exist_ok=True)
    for name, src in FILES.items():
        shutil.copy2(R / src, OUT / name)
    m = pd.read_parquet(OUT / "master_masked.parquet")
    lines = ["# PaddyCast data dictionary", "",
             f"master_masked / master_unmasked_7d: {len(m)} rows, one per district x season x "
             "lead_weeks; identical rows and columns in both. Block suffix _b1.._b5 = season "
             "time blocks; blocks after the cutoff are NaN.", "",
             "| column | dtype | non-null % | meaning |", "|---|---|---|---|"]
    for c in m.columns:
        lines.append(f"| {c} | {m[c].dtype} | {100 * m[c].notna().mean():.0f} | {describe(c)} |")
    (OUT / "data_dictionary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {"title": "PaddyCast masked master tables", "id": f"{user}/paddycast-masked",
            "licenses": [{"name": "other"}]}
    (OUT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("ready:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()