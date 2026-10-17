"""Season features from district time series + availability-checked yield lags.

Everything a row sees is restricted to information available at its FORECAST DATE:
    forecast_date = harvest_ref - lead_weeks
    harvest_ref   = nominal anchor (config season_calendar) + n_blocks * block_weeks
- Satellite/climate: a CHIRPS pentad counts only if its end (+ latency) <= forecast_date;
  a MODIS 16-day composite only if its end (start + 16 d, + latency) <= forecast_date.
- Anomalies: value minus the mean of the SAME district-season in strictly PREVIOUS years
  (min_history years required, else NaN). Never uses the current or future years.
- Yield lags: a past label is usable only if its own harvest_ref + label_availability_lag
  <= forecast_date (config data.label_availability_lag_months, an ASSUMPTION).
`shift_weeks` moves the growth-stage windows (sensitivity check) but not the forecast date.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MODIS_PERIOD_DAYS = 16


@dataclass(frozen=True)
class Calendar:
    anchors: dict          # season -> (month, day)
    block_days: int
    n_blocks: int
    pre_blocks: int
    lag_months: int
    latency_days: dict     # source -> days after period end before data are usable

    @classmethod
    def from_cfg(cls, cfg: dict) -> Calendar:
        f = cfg["features"]
        return cls(
            anchors={s: (v["anchor_month"], v["anchor_day"])
                     for s, v in f["season_calendar"].items()},
            block_days=7 * f["window_block_weeks"], n_blocks=f["n_blocks"],
            pre_blocks=f["pre_season_blocks"],
            lag_months=cfg["data"]["label_availability_lag_months"],
            latency_days=f.get("data_latency_days", {"chirps": 0, "modis": 0}))

    def anchor(self, season: str, season_year: int) -> pd.Timestamp:
        m, d = self.anchors[season]
        return pd.Timestamp(year=int(season_year), month=m, day=d)

    def harvest_ref(self, season: str, season_year: int) -> pd.Timestamp:
        return self.anchor(season, season_year) + pd.Timedelta(days=self.n_blocks * self.block_days)

    def forecast_date(self, season: str, season_year: int, lead_weeks: int) -> pd.Timestamp:
        return self.harvest_ref(season, season_year) - pd.Timedelta(weeks=lead_weeks)

    def blocks(self, season: str, season_year: int, shift_weeks: int = 0) -> list[tuple]:
        """[(name, start, end_exclusive)] for pre-season and growth-stage blocks."""
        a = self.anchor(season, season_year) + pd.Timedelta(weeks=shift_weeks)
        out = []
        for k in range(self.pre_blocks, 0, -1):
            s = a - pd.Timedelta(days=k * self.block_days)
            out.append((f"pre{k}", s, s + pd.Timedelta(days=self.block_days)))
        for k in range(self.n_blocks):
            s = a + pd.Timedelta(days=k * self.block_days)
            out.append((f"b{k + 1}", s, s + pd.Timedelta(days=self.block_days)))
        return out


# ------------------------------------------------------------------ input series
def chirps_daily(chirps: pd.DataFrame) -> pd.DataFrame:
    """Pentad totals -> daily rate per district (exact block sums for any window).
    Input columns: adm2_pcode, start, end, precipitation (mm per pentad)."""
    c = chirps.copy()
    if "precipitation" not in c and "mean" in c:   # EE names a single-reducer output 'mean'
        c = c.rename(columns={"mean": "precipitation"})
    c["start"] = pd.to_datetime(c["start"])
    c["end"] = pd.to_datetime(c["end"])
    c["ndays"] = (c["end"] - c["start"]).dt.days.clip(lower=1)
    c["rate"] = c["precipitation"] / c["ndays"]
    rows = c.loc[c.index.repeat(c["ndays"])].copy()
    rows["date"] = rows["start"] + pd.to_timedelta(rows.groupby(level=0).cumcount(), unit="D")
    rows["pentad_end"] = rows["end"]
    return rows[["adm2_pcode", "date", "pentad_end", "rate"]].reset_index(drop=True)


def modis_clean(modis: pd.DataFrame, static: pd.DataFrame | None) -> pd.DataFrame:
    m = modis.copy()
    m["date"] = pd.to_datetime(m["date"])
    m["end"] = m["date"] + pd.Timedelta(days=MODIS_PERIOD_DAYS)
    m["center"] = m["date"] + pd.Timedelta(days=MODIS_PERIOD_DAYS // 2)
    if static is not None and "crop" in static:
        m = m.merge(static[["adm2_pcode", "crop"]], on="adm2_pcode", how="left")
        m["valid_frac"] = m["NDVI_count"] / m["crop"]
    return m


# ------------------------------------------------------------------ season aggregation
def rain_features(rain: pd.DataFrame, pcode: str, season: str, sy: int, cal: Calendar,
                  fdate: pd.Timestamp, shift: int) -> dict:
    lat = pd.Timedelta(days=cal.latency_days.get("chirps", 0))
    r = rain[rain.pentad_end + lat <= fdate]          # rain: this district's rows only
    out, total, seen = {}, 0.0, False
    for name, s, e in cal.blocks(season, sy, shift):
        x = r[(r.date >= s) & (r.date < e)]
        val = float(x.rate.sum()) if len(x) else np.nan
        out[f"rain_{name}"] = val
        if name.startswith("b") and len(x):
            total += val
            seen = True
    out["rain_season"] = total if seen else np.nan
    return out


def modis_features(modis: pd.DataFrame, pcode: str, season: str, sy: int, cal: Calendar,
                   fdate: pd.Timestamp, shift: int) -> dict:
    lat = pd.Timedelta(days=cal.latency_days.get("modis", 0))
    m = modis[modis.end + lat <= fdate]               # modis: this district's rows only
    out = {}
    blocks = [b for b in cal.blocks(season, sy, shift) if b[0].startswith("b")]
    for name, s, e in blocks:
        x = m[(m.center >= s) & (m.center < e)]
        for v in ("NDVI", "EVI"):
            out[f"{v.lower()}_{name}"] = float(x[f"{v}_mean"].mean()) if len(x) else np.nan
        if "valid_frac" in m:
            out[f"modis_valid_{name}"] = float(x.valid_frac.mean()) if len(x) else np.nan
    s0, e_last = blocks[0][1], blocks[-1][2]
    x = m[(m.center >= s0) & (m.center < e_last)]
    out["ndvi_max"] = float(x.NDVI_mean.max()) if len(x) else np.nan
    out["evi_max"] = float(x.EVI_mean.max()) if len(x) else np.nan
    out["ndvi_auc"] = float(x.NDVI_mean.sum()) if len(x) else np.nan   # sum over composites
    return out


# ------------------------------------------------------------------ yield lags
def label_features(labels: pd.DataFrame, district: str, season: str, sy: int, cal: Calendar,
                   fdate: pd.Timestamp) -> dict:
    """labels: district, season, season_year, yield_kg_ha, sown_ha, harvested_ha (all years)."""
    lag = pd.DateOffset(months=cal.lag_months)
    d = labels[labels.district == district]
    avail = d[[cal.harvest_ref(s, y) + lag <= fdate
               for s, y in zip(d.season, d.season_year, strict=True)]]
    same = avail[avail.season == season].sort_values("season_year")
    prev = same[same.season_year == sy - 1]
    out = {
        "y_prev_same": float(prev.yield_kg_ha.iloc[0]) if len(prev) else np.nan,
        "y_mean_same": float(same.yield_kg_ha.mean()) if len(same) else np.nan,
        "y_last5_same": float(same.yield_kg_ha.tail(5).mean()) if len(same) >= 1 else np.nan,
        "n_label_hist_same": int(len(same)),
        "sown_prev_same": float(prev.sown_ha.iloc[0]) if len(prev) else np.nan,
        "harv_ratio_prev_same": float(prev.harvested_ha.iloc[0] / prev.sown_ha.iloc[0])
        if len(prev) and prev.sown_ha.iloc[0] else np.nan,
    }
    last10 = same.tail(10)
    if len(last10) >= 5:
        b = np.polyfit(last10.season_year, last10.yield_kg_ha, 1)
        out["y_trend_same"] = float(np.polyval(b, sy))
    else:
        out["y_trend_same"] = np.nan
    if len(avail):
        last = avail.assign(o=[cal.harvest_ref(s, y) for s, y in
                               zip(avail.season, avail.season_year, strict=True)]).sort_values("o")
        out["y_last_avail"] = float(last.yield_kg_ha.iloc[-1])
        out["y_last_avail_age_days"] = int((fdate - last.o.iloc[-1]).days)
    else:
        out["y_last_avail"], out["y_last_avail_age_days"] = np.nan, np.nan
    return out


def add_past_anomalies(df: pd.DataFrame, cols: list[str], min_history: int = 3) -> pd.DataFrame:
    """col_anom = col - mean(col) of the same district & season in strictly earlier years."""
    df = df.sort_values(["district", "season", "season_year"]).copy()
    for c in cols:
        g = df.groupby(["district", "season"])[c]
        past_mean = g.transform(lambda s: s.shift(1).expanding(min_periods=min_history).mean())
        df[f"{c}_anom"] = df[c] - past_mean
    return df


def build_season_features(targets: pd.DataFrame, labels: pd.DataFrame, rain: pd.DataFrame | None,
                          modis: pd.DataFrame | None, cal: Calendar, lead_weeks: int = 0,
                          shift_weeks: int = 0) -> pd.DataFrame:
    """targets: rows (district, adm2_pcode, season, season_year). Pass the FULL grid of
    seasons with climate data (not only rows with yields) so past-only anomalies have history;
    filter to modelling rows afterwards."""
    rain_by = dict(tuple(rain.groupby("adm2_pcode"))) if rain is not None else {}
    modis_by = dict(tuple(modis.groupby("adm2_pcode"))) if modis is not None else {}
    labels_by = dict(tuple(labels.groupby("district")))
    empty_labels = labels.iloc[0:0]
    recs = []
    for r in targets.itertuples(index=False):
        fdate = cal.forecast_date(r.season, r.season_year, lead_weeks)
        rec = {"district": r.district, "season": r.season, "season_year": r.season_year,
               "forecast_date": fdate, "lead_weeks": lead_weeks, "shift_weeks": shift_weeks}
        rec.update(label_features(labels_by.get(r.district, empty_labels), r.district, r.season,
                                  r.season_year, cal, fdate))
        if rain is not None and r.adm2_pcode in rain_by:
            rec.update(rain_features(rain_by[r.adm2_pcode], r.adm2_pcode, r.season, r.season_year,
                                     cal, fdate, shift_weeks))
        if modis is not None and r.adm2_pcode in modis_by:
            rec.update(modis_features(modis_by[r.adm2_pcode], r.adm2_pcode, r.season,
                                      r.season_year, cal, fdate, shift_weeks))
        recs.append(rec)
    out = pd.DataFrame(recs)
    climate = [c for c in out.columns if c.startswith(("rain_", "ndvi_", "evi_"))]
    return add_past_anomalies(out, climate)
