"""Tests for the DCS parser. Unit tests always run; real-file tests skip if data/raw is absent."""

import pandas as pd
import pytest

from src.config import REPO_ROOT
from src.data.dcs import (
    PROD_MAHA_LABEL_ORDER,
    STANDARD_ORDER,
    build,
    norm_district,
    repair_production_order,
    season_fields,
)

RAW = REPO_ROOT / "data" / "raw" / "dcs"
HAS_RAW = (RAW / "Average_Yeild-Maha_Season-1979-2025.xlsx").exists()


def test_season_convention_maha_start_year_and_order():
    m = season_fields("Maha", "2021/2022")
    y = season_fields("Yala", 2022)
    assert m["season_year"] == 2021 and m["harvest_year"] == 2022
    assert m["season_label"] == "2021/22 Maha"
    # Maha 2021/22 (harvest early 2022) comes before Yala 2022 (harvest late 2022)
    assert m["season_order"] < y["season_order"]
    assert y["season_order"] < season_fields("Maha", "2022/2023")["season_order"]


def test_district_spellings_normalise():
    for raw in ["KILLINOCHCHI", "KILINOCHCHIYA", "kilinochchi"]:
        assert norm_district(raw) == "KILINOCHCHI"
    assert norm_district("MULATIVU") == norm_district("MULLATIVU") == "MULLAITIVU"
    assert norm_district("MAHAWELI  'H'") == "MAHAWELI H"
    assert norm_district("NUWARAELIYA") == "NUWARA ELIYA"


def test_production_reorder_repairs_SYNTHETIC_shuffled_rows():
    """SYNTHETIC: production values written in standard order under shuffled labels."""
    rows = []
    for i, d in enumerate(STANDARD_ORDER):
        rows.append({"district": d, "season": "Maha", "label": "2000/2001",
                     "yield_kg_ha": 4000.0, "harvested_ha": 1000.0 * (i + 1)})
    t = pd.DataFrame(rows)
    true_kt = {d: 4000 * 1000 * (i + 1) * 0.9 / 1e6 for i, d in enumerate(STANDARD_ORDER)}
    shuffled = dict(zip(PROD_MAHA_LABEL_ORDER, [true_kt[d] for d in STANDARD_ORDER], strict=True))
    t["production_kt"] = t.district.map(shuffled)
    rep: list[str] = []
    fixed = repair_production_order(t, rep).set_index("district").production_kt
    assert all(abs(fixed[d] - true_kt[d]) < 1e-9 for d in STANDARD_ORDER)
    assert "2000/2001" in rep[-1]


@pytest.mark.skipif(not HAS_RAW, reason="raw DCS files not present")
def test_real_files_parse_and_match_pdf_2022_23_maha():
    t, rep = build(RAW)
    assert not t.duplicated(["district", "season", "season_year"]).any()
    assert any("'2000/2021'" in r and "2020/2021" in r for r in rep)
    assert any("'198'" in r for r in rep)
    x = t[(t.season == "Maha") & (t.season_year == 2022)].set_index("district")
    # values read from the DCS 2022/2023 Maha metric PDF
    assert x.loc["HAMBANTOTA", "yield_kg_ha"] == 5468
    assert abs(x.loc["HAMBANTOTA", "production_t"] - 178137) < 10
    assert abs(x.loc["AMPARA", "production_t"] - 318517) < 10
    assert x.loc["ANURADHAPURA", "harvested_ha"] == 117357
    assert set(t[t.unit_type == "district"].district) == set(STANDARD_ORDER[:25])
