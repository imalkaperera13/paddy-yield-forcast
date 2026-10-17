"""Step C1: DCS raw files -> tidy yield table, district map, sanity checks, missing heatmap.

    python -m src.data.build_yield

Outputs
  data/interim/dcs_all_units.csv     every parsed row (districts, special areas, national)
  data/interim/dcs_yield_tidy.csv    modelling units only (25 districts), rows with yield
  configs/district_map.csv           DCS spellings -> HDX COD-AB adm2 name + P-code
  results/tables/dcs_sanity_flags.csv, results/tables/dcs_missing_counts.csv
  results/tables/dcs_parse_report.txt
  results/figures/dcs_missing_heatmap.png
Nothing is imputed: missing yield rows are dropped and counted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import openpyxl  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from src.config import REPO_ROOT, load_config  # noqa: E402
from src.data.dcs import DISTRICT_ALIASES, build, norm_district  # noqa: E402

# First seasons reported WITH Mahaweli H / Udawalawe merged into host districts
# (footnote in every DCS series file).
MERGE_FROM = {"Maha": 2017, "Yala": 2018}

INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
PRESENT, MISSING = "#2a78d6", "#e4e3de"


def district_map(boundaries_xlsx: Path) -> pd.DataFrame:
    ws = openpyxl.load_workbook(boundaries_xlsx, read_only=True, data_only=True)["lka_admin2"]
    rows = list(ws.iter_rows(values_only=True))
    hdr = rows[0]
    adm = pd.DataFrame(rows[1:], columns=hdr)[["adm2_name", "adm2_pcode", "adm1_name"]]
    adm["key"] = adm.adm2_name.str.upper()
    spellings = set(DISTRICT_ALIASES) | set(adm.key)
    out = []
    for s in sorted(spellings):
        k = norm_district(s)
        m = adm[adm.key == k]
        if len(m):
            out.append({"dcs_spelling": s, "district": k, "adm2_name": m.adm2_name.iloc[0],
                        "adm2_pcode": m.adm2_pcode.iloc[0], "province": m.adm1_name.iloc[0]})
    return pd.DataFrame(out)


def sanity_flags(d: pd.DataFrame) -> pd.DataFrame:
    flags = []

    def add(mask, check, detail):
        for _, r in d[mask].iterrows():
            flags.append({"district": r.district, "season_label": r.season_label, "check": check,
                          "detail": detail(r)})

    add(d.yield_kg_ha.notna() & ((d.yield_kg_ha < 1000) | (d.yield_kg_ha > 7500)),
        "yield outside 1000-7500 kg/ha", lambda r: f"{r.yield_kg_ha:.0f}")
    add(d.harvested_ha > d.sown_ha * 1.001, "harvested > sown",
        lambda r: f"{r.harvested_ha:.0f} > {r.sown_ha:.0f}")
    ratio = d.production_t / (d.yield_kg_ha * d.harvested_ha / 1000)
    d = d.assign(_ratio=ratio)
    add((d._ratio < 0.6) | (d._ratio > 1.1), "production / (yield x harvested) outside 0.6-1.1",
        lambda r: f"{r._ratio:.2f}")
    # robust outlier within district-season: |x - median| / (1.4826 MAD) > 3.5
    g = d.groupby(["district", "season"]).yield_kg_ha
    med, mad = g.transform("median"), g.transform(lambda x: (x - x.median()).abs().median())
    z = (d.yield_kg_ha - med) / (1.4826 * mad)
    d = d.assign(_z=z)
    add(d._z.abs() > 3.5, "robust outlier (|z|>3.5 within district-season)",
        lambda r: f"yield {r.yield_kg_ha:.0f}, z={r._z:.1f}")
    return pd.DataFrame(flags)


def national_check(allu: pd.DataFrame) -> pd.DataFrame:
    """Area-weighted mean of all reported units vs the published SRI LANKA yield."""
    units = allu[allu.unit_type.isin(["district", "special_area"])].dropna(
        subset=["yield_kg_ha", "harvested_ha"])
    w = units.groupby(["season", "season_year"]).apply(
        lambda g: np.average(g.yield_kg_ha, weights=g.harvested_ha), include_groups=False)
    nat = allu[allu.unit_type == "national"].set_index(["season", "season_year"]).yield_kg_ha
    out = pd.DataFrame({"weighted_units": w, "published_national": nat}).dropna()
    out["diff_%"] = (out.weighted_units / out.published_national - 1) * 100
    return out.reset_index()


def missing_heatmap(d: pd.DataFrame, path: Path, dpi: int) -> None:
    """d: district rows after the zero-yield rule (NaN = missing)."""
    order = sorted(d.district.unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.2), facecolor=SURFACE,
                             gridspec_kw={"wspace": 0.08})
    cmap = ListedColormap([MISSING, PRESENT])
    for ax, season in zip(axes, ["Maha", "Yala"], strict=True):
        p = (d[d.season == season].pivot(index="district", columns="season_year",
                                         values="yield_kg_ha").reindex(order))
        ax.imshow(p.notna().to_numpy().astype(int), aspect="auto", cmap=cmap, vmin=0, vmax=1,
                  interpolation="nearest")
        # 2px-ish surface gaps between cells
        ax.set_xticks(np.arange(-.5, p.shape[1]), minor=True)
        ax.set_yticks(np.arange(-.5, p.shape[0]), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=1.2)
        ax.tick_params(which="minor", length=0)
        years = list(p.columns)
        ticks = [i for i, y in enumerate(years) if y % 5 == 0]
        ax.set_xticks(ticks, [str(years[i]) for i in ticks], color=MUTED, fontsize=8)
        ax.set_yticks(range(len(order)), [o.title() for o in order] if season == "Maha" else [],
                      color=INK, fontsize=8)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        n_miss = int(p.isna().sum().sum())
        label = "Maha (labelled by start year)" if season == "Maha" else "Yala"
        ax.set_title(f"{label} - {n_miss} missing of {p.size}", loc="left", fontsize=10,
                     color=INK)
    fig.suptitle("DCS district paddy yield: which district-seasons have a published value",
                 x=0.125, ha="left", fontsize=12, color=INK)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=PRESENT),
                        plt.Rectangle((0, 0), 1, 1, color=MISSING)],
               labels=["yield published", "missing / not reported"], loc="lower left",
               bbox_to_anchor=(0.125, -0.005), ncol=2, frameon=False, fontsize=9)
    fig.text(0.125, -0.04, "Source: Department of Census and Statistics, Sri Lanka "
             "(downloaded 2026-10-17). Missing values are dropped, never imputed.",
             fontsize=8, color=MUTED)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    raw = REPO_ROOT / cfg["paths"]["raw"]
    interim = REPO_ROOT / cfg["paths"]["interim"]
    tables = REPO_ROOT / cfg["paths"]["tables"]
    figs = REPO_ROOT / cfg["paths"]["figures"]
    for p in (interim, tables, figs):
        p.mkdir(parents=True, exist_ok=True)

    allu, report = build(raw / "dcs")
    allu["mahaweli_merged"] = [int(y >= MERGE_FROM[s])
                               for s, y in zip(allu.season, allu.season_year, strict=True)]
    allu.to_csv(interim / "dcs_all_units.csv", index=False)

    dmap = district_map(raw / "boundaries" / "lka_admin_boundaries.xlsx")
    dmap.to_csv(REPO_ROOT / "configs" / "district_map.csv", index=False)

    d = allu[allu.unit_type == "district"].copy()
    # A published yield of 0 is a 'not reported' placeholder (e.g. Ampara 1990 Yala: 0 kg/ha
    # but 125,100 t produced). Treated as missing - dropped, not imputed.
    zero = d.yield_kg_ha == 0
    report.append(f"yield == 0 treated as not reported: {int(zero.sum())} rows "
                  f"({', '.join(sorted(set(d.loc[zero, 'season_label'])))})")
    d.loc[zero, "yield_kg_ha"] = np.nan
    unmapped = sorted(set(d.district) - set(dmap.district))
    report.append(f"districts not matched to boundaries: {unmapped or 'none'}")
    d = d.merge(dmap.drop_duplicates("district")[["district", "adm2_pcode"]], on="district",
                how="left")

    miss = (d.assign(missing=d.yield_kg_ha.isna())
            .groupby(["season", "district"]).missing.sum().unstack(0).fillna(0).astype(int))
    miss.loc["TOTAL"] = miss.sum()
    miss.to_csv(tables / "dcs_missing_counts.csv")
    n_before = len(d)
    tidy = d.dropna(subset=["yield_kg_ha"]).reset_index(drop=True)
    report.append(f"district rows: {n_before}; dropped (no yield): {n_before - len(tidy)}; "
                  f"kept: {len(tidy)}")
    for s in ["Maha", "Yala"]:
        x = tidy[tidy.season == s]
        report.append(f"  {s}: {len(x)} rows, seasons {x.season_year.min()}-"
                      f"{x.season_year.max()}, {x.district.nunique()} districts")
    tidy.to_csv(interim / "dcs_yield_tidy.csv", index=False)

    flags = sanity_flags(tidy)
    flags.to_csv(tables / "dcs_sanity_flags.csv", index=False)
    flagged = set(zip(flags.district, flags.season_label, strict=True)) if len(flags) else set()
    tidy["sanity_flag"] = [int((d_, s_) in flagged)
                           for d_, s_ in zip(tidy.district, tidy.season_label, strict=True)]
    tidy.to_csv(interim / "dcs_yield_tidy.csv", index=False)
    report.append(f"rows with >=1 sanity flag: {int(tidy.sanity_flag.sum())}")
    report.append("sanity flags by check: "
                  + (flags.check.value_counts().to_dict().__repr__() if len(flags) else "none"))
    nat = national_check(allu)
    nat.to_csv(tables / "dcs_national_check.csv", index=False)
    report.append(f"national check |diff| %: median {nat['diff_%'].abs().median():.2f}, "
                  f"max {nat['diff_%'].abs().max():.2f}")

    missing_heatmap(d, figs / "dcs_missing_heatmap.png", cfg["figures"]["dpi"])
    (tables / "dcs_parse_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
