"""Step 3a: leakage-free STATIC paddy mask for feature extraction (Earth Engine).

Why: the seasonal mask (src/data/paddy_mask.py) for season t uses season t's own flooding,
green-up and harvest, all of which happen after the forecast cutoff. Using it for features
would leak. The static mask uses only the source seasons in config
features.paddy_mask.static.years (2000-2008), all strictly before the first forward-chaining
test season (Maha 2009/10), so no test season enters the mask.

    # 1. detection frequency per district (how many of the 9 source seasons each pixel was
    #    detected), one request per season type
    .venv/Scripts/python -m src.data.static_mask freq
    # 2. choose k (detected in >= k of 9) from the frequency table and DCS sown 2000-2008
    .venv/Scripts/python -m src.data.static_mask choose
    # 3. export the two masks (Maha, Yala) as Earth Engine assets (mean pyramids, so coarse
    #    datasets such as CHIRPS/ERA5 read them as an area fraction); waits for the tasks
    .venv/Scripts/python -m src.data.static_mask export
    # 4. static mask area vs DCS sown extent per district and season
    .venv/Scripts/python -m src.data.static_mask validate

Rule for k (no yields used): for each season type, pick the k in 1..N whose pooled static
mask area over the selected districts is closest to the pooled mean DCS sown extent of the
same season over the source years only. Ties go to the larger k (more conservative).
Set features.paddy_mask.static.k to an integer to override.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from src.config import REPO_ROOT, load_config
from src.data.ee_extract import ee_init, get_with_retry, log
from src.data.paddy_mask import GEOJSON, build_mask, selected_codes

MASK_DIR = REPO_ROOT / "data" / "interim" / "ee" / "mask"
FREQ = MASK_DIR / "static_freq.csv"
KFILE = MASK_DIR / "static_k.json"
AREA = MASK_DIR / "static_area.csv"
VALID = REPO_ROOT / "results" / "tables" / "mask_static_validation.csv"
DCS = REPO_ROOT / "data" / "interim" / "dcs_yield_tidy.csv"
SCALE_M = 250
TILE_SCALE = 8


# ---------------------------------------------------------------- pure Python (tested)
def season_order(season: str, season_year: int) -> int:
    """Same convention as the data: 2*harvest_year + (0 Maha, 1 Yala)."""
    harvest_year = season_year + 1 if season == "Maha" else season_year
    return 2 * harvest_year + (0 if season == "Maha" else 1)


def static_source_seasons(cfg: dict, season: str) -> list[tuple[str, int]]:
    """Source seasons for the static mask of one season type; all before the first test season."""
    st = cfg["features"]["paddy_mask"]["static"]
    y0, y1 = st["years"]
    ft = st["first_test"]
    cutoff = season_order(ft["season"], ft["season_year"])
    out = [(season, y) for y in range(y0, y1 + 1)]
    bad = [s for s in out if season_order(*s) >= cutoff]
    if bad:
        raise ValueError(f"static mask source seasons not before first test season: {bad}")
    return out


def choose_k(freq: pd.DataFrame, dcs: pd.DataFrame, cfg: dict) -> dict:
    """freq: adm2_pcode, season, count, ha  (area detected in exactly `count` source seasons).
    dcs: DCS tidy table. Uses ONLY sown_ha of the source years; never yields."""
    st = cfg["features"]["paddy_mask"]["static"]
    y0, y1 = st["years"]
    fixed = st.get("k", "auto")
    d = dcs[dcs["season_year"].between(y0, y1)][["adm2_pcode", "season", "season_year", "sown_ha"]]
    out = {}
    for season, g in freq.groupby("season"):
        codes = sorted(g["adm2_pcode"].unique())
        n_src = len(static_source_seasons(cfg, season))
        target = (d[(d.season == season) & d.adm2_pcode.isin(codes)]
                  .groupby("adm2_pcode")["sown_ha"].mean().sum())
        rows = []
        for k in range(1, n_src + 1):
            area = g[g["count"] >= k]["ha"].sum()
            rows.append({"k": k, "static_ha": float(area),
                         "ratio_vs_dcs": float(area / target) if target else None})
        table = pd.DataFrame(rows)
        if fixed != "auto":
            k = int(fixed)
        else:
            err = (table["ratio_vs_dcs"] - 1).abs()
            k = int(table.loc[err == err.min(), "k"].max())
        out[season] = {"k": k, "n_source_seasons": n_src,
                       "dcs_sown_mean_ha_source_years": float(target),
                       "rule": "auto: pooled area closest to pooled DCS sown (source years)"
                       if fixed == "auto" else "fixed in config",
                       "table": table.to_dict(orient="records")}
    return out


# ---------------------------------------------------------------- Earth Engine
def _fc(ee, codes):
    gj = json.loads(GEOJSON.read_text(encoding="utf-8"))
    gj["features"] = [f for f in gj["features"] if f["properties"]["adm2_pcode"] in codes]
    for f in gj["features"]:
        f["properties"] = {"adm2_pcode": f["properties"]["adm2_pcode"]}
    return ee.FeatureCollection(gj)


def detection_count(ee, cfg: dict, season: str):
    """Number of source seasons in which each 250 m pixel passed the strict seasonal rules."""
    imgs, grid = [], None
    for s, y in static_source_seasons(cfg, season):
        strict, _, _, _, grid = build_mask(ee, cfg, s, y)
        imgs.append(strict.toInt())
    return ee.ImageCollection(imgs).sum().rename("count"), grid


def cmd_freq(cfg: dict) -> None:
    ee = ee_init(cfg)
    codes = selected_codes(cfg)
    fc = _fc(ee, codes)
    rows = []
    for season in ("Maha", "Yala"):
        cnt, grid = detection_count(ee, cfg, season)
        img = ee.Image.pixelArea().divide(1e4).rename("ha").addBands(cnt)
        red = ee.Reducer.sum().group(groupField=1, groupName="count")
        res = img.reduceRegions(collection=fc, reducer=red, scale=SCALE_M, crs=grid.crs(),
                                tileScale=TILE_SCALE)
        info = get_with_retry(res.map(lambda f: ee.Feature(None, f.toDictionary())),
                              f"freq {season}")
        for f in info["features"]:
            p = f["properties"]
            for grp in p.get("groups", []):
                rows.append({"adm2_pcode": p["adm2_pcode"], "season": season,
                             "count": int(grp["count"]), "ha": float(grp["sum"])})
        log(f"freq {season}: done")
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["season", "adm2_pcode", "count"]).to_csv(FREQ, index=False)
    log(f"freq: wrote {FREQ.relative_to(REPO_ROOT)}")


def cmd_choose(cfg: dict) -> None:
    freq = pd.read_csv(FREQ)
    res = choose_k(freq, pd.read_csv(DCS), cfg)
    KFILE.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for season, r in res.items():
        print(f"\n{season}: k = {r['k']} of {r['n_source_seasons']} ({r['rule']}); "
              f"pooled DCS sown (source years) {r['dcs_sown_mean_ha_source_years']:,.0f} ha")
        print(pd.DataFrame(r["table"]).round(3).to_string(index=False))
    print(f"\nwrote {KFILE.relative_to(REPO_ROOT)}")


def asset_id(cfg: dict, season: str) -> str:
    st = cfg["features"]["paddy_mask"]["static"]
    y0, y1 = st["years"]
    return f"{st['asset_root']}/static_mask_{season.lower()}_{y0}_{y1}"


def static_mask_image(ee, cfg: dict, season: str):
    """The exported static mask (0/1 on the 250 m grid; mean pyramids give area fractions)."""
    return ee.Image(asset_id(cfg, season)).select("mask")


def cmd_export(cfg: dict, force: bool) -> None:
    ee = ee_init(cfg)
    k_all = json.loads(KFILE.read_text(encoding="utf-8"))
    codes = selected_codes(cfg)
    region = _fc(ee, codes).geometry().bounds()
    root = cfg["features"]["paddy_mask"]["static"]["asset_root"]
    try:
        ee.data.getAsset(root)
    except Exception:
        ee.data.createAsset({"type": "FOLDER"}, root)
        log(f"export: created asset folder {root}")
    tasks = []
    for season in ("Maha", "Yala"):
        aid = asset_id(cfg, season)
        exists = True
        try:
            ee.data.getAsset(aid)
        except Exception:
            exists = False
        if exists and not force:
            log(f"export {season}: {aid} exists, skipped (use --force to replace)")
            continue
        if exists:
            ee.data.deleteAsset(aid)
        k = int(k_all[season]["k"])
        cnt, grid = detection_count(ee, cfg, season)
        pinfo = grid.getInfo()
        src = [f"{s} {y}" for s, y in static_source_seasons(cfg, season)]
        img = (cnt.gte(k).unmask(0).rename("mask").toByte()
               .set({"season": season, "k": k, "source_seasons": ",".join(src),
                     "method": "strict seasonal rules, detected in >= k source seasons"}))
        task = ee.batch.Export.image.toAsset(
            image=img, description=f"paddycast_static_{season.lower()}", assetId=aid,
            region=region, crs=pinfo["crs"], crsTransform=pinfo["transform"],
            pyramidingPolicy={".default": "mean"}, maxPixels=1e13)
        task.start()
        tasks.append((season, task))
        log(f"export {season}: started task for {aid} (k={k})")
    t0 = time.time()
    while tasks:
        time.sleep(30)
        left = []
        for season, task in tasks:
            st = task.status()["state"]
            if st == "COMPLETED":
                log(f"export {season}: COMPLETED")
            elif st in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"export {season}: {st} {task.status().get('error_message')}")
            else:
                left.append((season, task))
        tasks = left
        if tasks and time.time() - t0 > 3600:
            raise RuntimeError("export: tasks still running after 60 min; check the EE Tasks page")


def cmd_validate(cfg: dict) -> None:
    ee = ee_init(cfg)
    codes = selected_codes(cfg)
    fc = _fc(ee, codes)
    grid = ee.ImageCollection("MODIS/061/MOD13Q1").first().select("NDVI").projection()
    rows = []
    for season in ("Maha", "Yala"):
        m = static_mask_image(ee, cfg, season)
        img = ee.Image.pixelArea().divide(1e4).multiply(m).rename("ha")
        res = img.reduceRegions(collection=fc, reducer=ee.Reducer.sum(), scale=SCALE_M,
                                crs=grid.crs(), tileScale=TILE_SCALE)
        info = get_with_retry(res.map(lambda f: ee.Feature(None, f.toDictionary())),
                              f"static area {season}")
        for f in info["features"]:
            rows.append({"adm2_pcode": f["properties"]["adm2_pcode"], "season": season,
                         "static_mask_ha": f["properties"].get("sum")})
    area = pd.DataFrame(rows)
    area.to_csv(AREA, index=False)

    k_all = json.loads(KFILE.read_text(encoding="utf-8"))
    y0, y1 = cfg["features"]["paddy_mask"]["static"]["years"]
    d = pd.read_csv(DCS)
    d = d[d.adm2_pcode.isin(codes) & d.sown_ha.notna() & (d.season_year >= y0)]
    m = d.merge(area, on=["adm2_pcode", "season"], how="left")
    m["ratio"] = m["static_mask_ha"] / m["sown_ha"]
    out = []
    for (dist, code, season), g in m.groupby(["district", "adm2_pcode", "season"]):
        src = g[g.season_year <= y1]
        oos = g[g.season_year > y1]
        out.append({
            "district": dist, "adm2_pcode": code, "season": season,
            "k": k_all[season]["k"],
            "static_mask_ha": float(g.static_mask_ha.iloc[0]),
            "years": f"{int(g.season_year.min())}-{int(g.season_year.max())}",
            "n_years": len(g),
            "dcs_sown_mean_ha": float(g.sown_ha.mean()),
            "ratio_mean": float(g.ratio.mean()),
            "ratio_min": float(g.ratio.min()),
            "ratio_max": float(g.ratio.max()),
            "ratio_mean_source_years": float(src.ratio.mean()) if len(src) else None,
            "ratio_mean_after_source": float(oos.ratio.mean()) if len(oos) else None,
        })
    v = pd.DataFrame(out)
    pooled = (v.groupby("season")[["static_mask_ha", "dcs_sown_mean_ha"]].sum()
              .assign(ratio_of_totals=lambda x: x.static_mask_ha / x.dcs_sown_mean_ha))
    VALID.parent.mkdir(parents=True, exist_ok=True)
    v.to_csv(VALID, index=False)
    print(v.round(3).to_string(index=False))
    print("\npooled (sum over districts):")
    print(pooled.round(3).to_string())
    print(f"\nwrote {VALID.relative_to(REPO_ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freq", "choose", "export", "validate"])
    ap.add_argument("--force", action="store_true", help="export: replace existing assets")
    a = ap.parse_args()
    cfg = load_config()
    {"freq": lambda: cmd_freq(cfg), "choose": lambda: cmd_choose(cfg),
     "export": lambda: cmd_export(cfg, a.force), "validate": lambda: cmd_validate(cfg)}[a.cmd]()


if __name__ == "__main__":
    main()
