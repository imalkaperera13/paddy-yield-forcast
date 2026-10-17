# Data card: master_table.parquet

Built by `src/data/build_master.py` (lead 0, nominal windows). Target: `yield_kg_ha`. **1266 rows**, 25 districts.

**Scope:** the modelling table starts where satellite and climate features exist (Maha 2000/01, Yala 2000). DCS yields from 1979–1999 are used ONLY to compute lagged-yield features and the baseline; they are never training rows.

| season | rows | first season_year | last season_year | districts |
|---|---|---|---|---|
| Maha | 647 | 2000 | 2025 | 25 |
| Yala | 619 | 2000 | 2025 | 24 |

## Feature coverage (non-missing share of rows)

| group | n features (incl. anomalies) | mean coverage % | worst feature % |
|---|---|---|---|
| lagged_yield | 9 | 99.8 | 99.4 |
| rainfall | 14 | 100.0 | 100.0 |
| modis | 31 | 99.9 | 99.7 |
| temp_radiation | 66 | 100.0 | 100.0 |

Anomalies are relative to the same district-season in PREVIOUS years only (at least 3 years). Yield lags respect the assumed 6-month publication lag.

## Diagnostic-only columns (never features)

`mahaweli_merged`, `mahaweli_affected`, `diag_post_merge_affected`, `sanity_flag` (59 rows flagged).

## Shock-season candidates (from the data)

Largest negative national area-weighted yield anomalies vs the previous 10 same seasons. 2010/11 Maha (January–February 2011 floods) was named in advance.

| season | season_year | national yield kg/ha | past-10 mean | anomaly % |
|---|---|---|---|---|
| Maha | 2021 | 2831 | 4382 | -35.4 |
| Yala | 2022 | 3405 | 4447 | -23.4 |
| Maha | 2022 | 3732 | 4231 | -11.8 |
| Maha | 2010 | 3686 | 4021 | -8.3 |
| Maha | 2023 | 3892 | 4185 | -7.0 |
| Maha | 2024 | 3915 | 4156 | -5.8 |
| Yala | 2023 | 4133 | 4378 | -5.6 |
| Yala | 2014 | 4115 | 4191 | -1.8 |
| Yala | 2017 | 4222 | 4253 | -0.7 |
| Yala | 2012 | 4088 | 4084 | 0.1 |

## Known limitations

See `results/EXPLAINED.md` → Limitations observed, and `docs/decisions.md`.
