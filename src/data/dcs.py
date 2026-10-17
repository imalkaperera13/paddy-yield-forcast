"""Parse the eight DCS district paddy series (1979-2025) into one tidy table.

Raw files are read-only. Layout quirks handled (all logged in the parse report):
- the header row is found by searching for the 'District' cell (row position varies);
- yield / extent files have Major / Minor / Rainfed / Total sub-columns -> 'Total' is used;
- production files have one column per season;
- the Production-Yala file has the header '198' for 1980 (typo, repaired by sequence);
- '-' and blanks are missing values; district spellings are normalised via DISTRICT_ALIASES.

Season-year convention (docs/decisions.md): see `season_fields`.
"""

from __future__ import annotations

import re
from pathlib import Path

import openpyxl
import pandas as pd

SERIES = {
    ("yield_kg_ha", "Maha"): "Average_Yeild-Maha_Season-1979-2025.xlsx",
    ("yield_kg_ha", "Yala"): "Average_Yeild-Yala_Season-1979-2025.xlsx",
    ("production_kt", "Maha"): "Production-Maha_Season-1979-2025.xlsx",
    ("production_kt", "Yala"): "Production-Yala_Season-1979-2025.xlsx",
    ("sown_ha", "Maha"): "Sown_Extent-Maha_Seasons-1979-2025.xlsx",
    ("sown_ha", "Yala"): "Sown_Extent-Yala_Seasons-1979-2025.xlsx",
    ("harvested_ha", "Maha"): "Harvested_Extent-Maha_Seasons-1979-2025.xlsx",
    ("harvested_ha", "Yala"): "Harvested_Extent-Yala_Seasons-1979-2025.xlsx",
}

DISTRICT_ALIASES = {
    "KILLINOCHCHI": "KILINOCHCHI", "KILINOCHCHIYA": "KILINOCHCHI",
    "MULATIVU": "MULLAITIVU", "MULLATIVU": "MULLAITIVU",
    "NUWARAELIYA": "NUWARA ELIYA",
    "MAHAWELI'H'": "MAHAWELI H", "MAHAWELI 'H'": "MAHAWELI H", "MAHAWELI  'H'": "MAHAWELI H",
}
SPECIAL_UNITS = {"MAHAWELI H", "UDAWALAWE"}      # irrigation areas, no admin polygon
NON_DISTRICT = {"SRI LANKA", "HIGHLAND PADDY"}   # national total / production-only row


def norm_district(name: str) -> str:
    n = re.sub(r"\s+", " ", str(name).strip().upper())
    n = DISTRICT_ALIASES.get(n, DISTRICT_ALIASES.get(n.replace(" ", ""), n))
    return DISTRICT_ALIASES.get(n, n)


def season_fields(season: str, label) -> dict:
    """Maha 'YYYY/YYYY+1' is labelled by its START year (planting Oct-Dec of YYYY,
    harvest Feb-Apr of YYYY+1). Yala 'YYYY' is planted Apr-May and harvested Aug-Sep of YYYY.
    season_order sorts seasons by harvest time: Maha Y (harvest early Y+1) comes before
    Yala Y+1 (harvest late Y+1)."""
    if season == "Maha":
        start = int(str(label).split("/")[0])
        harvest = start + 1
        order = harvest * 2 + 0
    else:
        start = harvest = int(label)
        order = harvest * 2 + 1
    return {"season_year": start, "harvest_year": harvest, "season_order": order,
            "season_label": f"{start}/{str(start + 1)[-2:]} Maha" if season == "Maha"
            else f"{start} Yala"}


def _cell_missing(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip() in {"", "-", "–", "n.a", "na"})


def parse_file(path: Path, variable: str, season: str, report: list[str]) -> pd.DataFrame:
    rows = list(openpyxl.load_workbook(path, data_only=True, read_only=True).active
                .iter_rows(values_only=True))
    hi = next(i for i, r in enumerate(rows)
              if any(isinstance(c, str) and c.strip().lower() == "district" for c in r))
    hdr = rows[hi]
    dcol = next(j for j, c in enumerate(hdr)
                if isinstance(c, str) and c.strip().lower() == "district")
    # find the sub-header row with 'Total' (extent / yield files), if any
    sub = None
    for k in (hi + 1, hi + 2):
        if k < len(rows) and any(isinstance(c, str) and c.strip().lower() == "total"
                                 for c in rows[k]):
            sub = k
            break
    year_cols: list[tuple[int, object]] = []
    for j, c in enumerate(hdr):
        if j <= dcol or c is None:
            continue
        if sub is None:
            year_cols.append((j, c))
        else:  # value column = the 'Total' column inside this year's block of 4
            tj = next(t for t in range(j, min(j + 4, len(rows[sub])))
                      if isinstance(rows[sub][t], str) and rows[sub][t].strip().lower() == "total")
            year_cols.append((tj, c))
    # repair typos in year labels by sequence (e.g. '198' between 1979 and 1981,
    # '2000/2021' between 2019/2020 and 2021/2022). Every label must be prev + 1.
    labels = []
    for idx, (_, lab) in enumerate(year_cols):
        s = str(lab).strip()
        ok = re.fullmatch(r"\d{4}/\d{4}", s) if season == "Maha" else re.fullmatch(r"\d{4}", s)
        if ok and season == "Maha":
            a, b = (int(x) for x in s.split("/"))
            ok = b == a + 1 and (not labels or a == int(labels[-1].split("/")[0]) + 1)
        elif ok and labels:
            ok = int(s) == int(labels[-1]) + 1
        if not ok:
            if not labels:
                raise ValueError(f"{path.name}: first season label '{s}' is invalid")
            prev = labels[-1]
            fixed = (f"{int(prev.split('/')[0]) + 1}/{int(prev.split('/')[1]) + 1}"
                     if season == "Maha" else str(int(prev) + 1))
            report.append(f"{path.name}: header '{s}' at position {idx} repaired to '{fixed}'")
            s = fixed
        labels.append(s)
    out = []
    first_data = (sub if sub is not None else hi) + 1
    for r in rows[first_data:]:
        name = r[dcol] if dcol < len(r) else None
        if name is None or (isinstance(name, str) and len(name) > 40):
            continue
        d = norm_district(name)
        for (j, _), lab in zip(year_cols, labels, strict=True):
            v = r[j] if j < len(r) else None
            out.append({"district": d, "season": season, "label": lab, variable:
                        None if _cell_missing(v) else float(v)})
    df = pd.DataFrame(out)
    kind = "Total sub-column" if sub else "single column"
    report.append(f"{path.name}: header row {hi + 1}, {kind}, {len(labels)} seasons "
                  f"{labels[0]}..{labels[-1]}, {df.district.nunique()} units")
    return df


# Production-Maha file: row labels are in this order, while for some seasons the VALUES
# follow the standard order used by every other file (verified against yield x harvested
# extent and against the DCS 2022/23 Maha PDF). See repair_production_order().
PROD_MAHA_LABEL_ORDER = [
    "COLOMBO", "GAMPAHA", "KALUTARA", "GALLE", "MATARA", "RATNAPURA", "KEGALLE", "KURUNEGALA",
    "PUTTALAM", "KANDY", "MATALE", "NUWARA ELIYA", "BADULLA", "MONARAGALA", "JAFFNA",
    "KILINOCHCHI", "VAVUNIYA", "MULLAITIVU", "MANNAR", "ANURADHAPURA", "POLONNARUWA",
    "TRINCOMALEE", "BATTICALOA", "AMPARA", "HAMBANTOTA", "UDAWALAWE", "MAHAWELI H"]
STANDARD_ORDER = [
    "COLOMBO", "GAMPAHA", "KALUTARA", "KANDY", "MATALE", "NUWARA ELIYA", "GALLE", "MATARA",
    "HAMBANTOTA", "JAFFNA", "MANNAR", "VAVUNIYA", "MULLAITIVU", "KILINOCHCHI", "BATTICALOA",
    "AMPARA", "TRINCOMALEE", "KURUNEGALA", "PUTTALAM", "ANURADHAPURA", "POLONNARUWA", "BADULLA",
    "MONARAGALA", "RATNAPURA", "KEGALLE", "UDAWALAWE", "MAHAWELI H"]


def _n_consistent(prod: pd.Series, implied: pd.Series) -> int:
    r = prod / implied
    return int(((r > 0.75) & (r < 1.05)).sum())


def repair_production_order(t: pd.DataFrame, report: list[str]) -> pd.DataFrame:
    """For each Maha season, keep the production row order (as labelled, or re-mapped to the
    standard order) that agrees with yield_kg_ha x harvested_ha for more districts. Yield is
    per NET hectare and extent is GROSS, so the consistent band is 0.75-1.05."""
    remap = dict(zip(PROD_MAHA_LABEL_ORDER, STANDARD_ORDER, strict=True))
    t = t.copy()
    fixed = []
    for y in sorted(t.loc[t.season == "Maha", "label"].unique()):
        m = (t.season == "Maha") & (t.label == y) & t.district.isin(remap)
        g = t[m].set_index("district")
        implied_kt = g.yield_kg_ha * g.harvested_ha / 1e6   # kg/ha * ha -> thousand tonnes
        as_lab = _n_consistent(g.production_kt, implied_kt)
        alt = pd.Series({remap[d]: g.production_kt.get(d) for d in g.index}).reindex(g.index)
        as_std = _n_consistent(alt, implied_kt)
        if as_std > as_lab:
            t.loc[m, "production_kt"] = t.loc[m, "district"].map(alt).to_numpy()
            fixed.append(f"{y} ({as_lab}->{as_std} consistent)")
    report.append("Production-Maha row order re-mapped to the standard district order for: "
                  + (", ".join(fixed) if fixed else "none"))
    return t


def build(raw_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    report: list[str] = []
    frames = {}
    for (var, season), fname in SERIES.items():
        df = parse_file(raw_dir / fname, var, season, report)
        frames.setdefault(season, []).append(df.set_index(["district", "season", "label"]))
    tidy = []
    for dfs in frames.values():
        tidy.append(pd.concat(dfs, axis=1).reset_index())
    t = pd.concat(tidy, ignore_index=True)
    for c in ["yield_kg_ha", "production_kt", "sown_ha", "harvested_ha"]:
        t[c] = pd.to_numeric(t[c])
    t = repair_production_order(t, report)
    sf = t.apply(lambda r: pd.Series(season_fields(r["season"], r["label"])), axis=1)
    t = pd.concat([t, sf], axis=1)
    t["unit_type"] = t["district"].map(
        lambda d: "national" if d == "SRI LANKA" else "highland" if d == "HIGHLAND PADDY"
        else "special_area" if d in SPECIAL_UNITS else "district")
    t["production_t"] = t["production_kt"] * 1000
    cols = ["district", "unit_type", "season", "season_label", "season_year", "harvest_year",
            "season_order", "yield_kg_ha", "production_t", "sown_ha", "harvested_ha"]
    return t[cols].sort_values(["season_order", "district"]).reset_index(drop=True), report
