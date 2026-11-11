# Data card: `data/`

The curated data our code writes and uses. `code/Protostar_Final_Pipeline.ipynb` writes every file except the three NOAA files, which `code/integration/noaa_station_metadata.py` writes. Rerunning them regenerates the files. The notebook reads the raw training data from Hugging Face at a pinned version, so no raw data needs to be uploaded.

## Files

The challenge dataset is [`CMU-OSPO/protostar-climate-train-to-2018`](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018), at commit `bbbdebdab2a18e00f97ef87bd3f35a71e222180b`.

| File | What it holds | Source | License | Years covered | How we prepared it | Used for |
|---|---|---|---|---|---|---|
| `labels_final.csv.gz` | One row per city, year and target: `official` value, value after filling missing days (`label_before_homogenisation`), station-change correction at 2018 (`homogenisation_adjustment`), value the forecasters read (`label_used_at_2018`), `source` (official or infilled) and `missing_days` | Challenge dataset, `annual_indices` and `daily` | CC0-1.0 | 1851–2018 | Missing days filled for Avg Temp and CDD (up to 30 per year) from the trailing 10-year climatology; station-change jumps corrected against the median of the 8 nearest neighbours | Forecast input |
| `level_shifts_at_2018.csv` | Detected station-change jumps for Avg Temp and CDD: `break_year`, `shift` (added to earlier years) and t-statistic `t` | Challenge dataset | CC0-1.0 | Jumps in 1955–2014 | Lasting steps in each city's difference from its neighbours' median (t ≥ 5, ≥ 0.6 °F or 120 °F-days, ≥ 5 years on each side) | Forecast input (via the correction) and analysis |
| `quality_flags.csv` | One row per city-year: missing days for daily high, low, precipitation and high/low pairs; how much infilling changed CDD and Avg Temp; whether each target passes the 10-missing-day rule (`strict_valid_*`); days in suspicious stuck-high runs | Challenge dataset, `daily` | CC0-1.0 | 1851–2018 | Counted from `daily`, and checked against the dataset's own `Miss` columns | Analysis (flags recorded, not removed) |
| `stuck_tmax_runs.csv` | Runs of at least 7 days with the same daily high; `expected_runs` is how many chance would produce, and `suspicious` marks unlikely runs | Challenge dataset, `daily` | CC0-1.0 | 1883–2018 | Runs found in `daily`; expected count by chance for that station and month | Analysis |
| `final_selection_tables.csv` | Backtest error for each candidate team rule, start years 1990–1999 and 2000–2011 | Our backtest of the challenge dataset | CC-BY-4.0 | Forecasts of 1981–2018 | Rolling-origin backtest, start years 1980–2011 | Model selection |
| `final_selected_methods.csv` | The team rule selected for each target | Our backtest | CC-BY-4.0 | – | Lowest era-balanced error | Model selection |
| `ensemble_backtest_1990_2011.csv.gz` | Backtest rows behind the ensemble's score: start year (`origin`), target, city index (position in the sorted list of station IDs), year, actual value, 30-year baseline (`clim30`) and each component's forecast | Our backtest | CC-BY-4.0 | Forecasts of 1991–2018 | Rolling-origin backtest, start years 1990–2011 | Model checking |
| `ensemble_components_2019_2025.csv` | For every submission row: `team`, `ridge`, `factor` and the submitted `ensemble` (0.50 / 0.25 / 0.25); an empty `ridge` or `factor` means the team value was used in its place | Our forecasts | CC-BY-4.0 | 2019–2025 (forecasts) | Each component run at origin 2018 | Output (the submission's breakdown) |
| `submission_team_only.csv` | The team's rules alone, in the format of `submission.csv` | Our forecasts | CC-BY-4.0 | 2019–2025 (forecasts) | Team component only | Comparison |
| `ghcnd_station_crosswalk.csv` | Each challenge station next to NOAA's GHCN-Daily entry: name, coordinates, elevation, network flags (`gsn`, `hcn_crn`), WMO ID and distance between the two locations (`offset_km`) | [NOAA GHCN-Daily station list](https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt) | Public domain (U.S. government) | Snapshot of 2026-11-10 (station metadata, not weather) | Joined on GHCN-Daily ID | Analysis only |
| `noaa_homr_events_to_2018.csv` | Dated relocations (distance and direction) and changes of temperature equipment for each station; `ncdc_stn_id` is NOAA's internal ID | [NOAA HOMR](https://www.ncei.noaa.gov/access/homr/) | Public domain (U.S. government) | Events 1871–2018 | Station histories requested by GHCN-Daily ID; events after 2018-12-31 dropped | Analysis only |
| `station_break_attribution.csv` | Each jump in `level_shifts_at_2018.csv`, with any `city_info` station change, NOAA relocation or NOAA equipment change within one year; `explained_by` is `city_info`, `noaa_homr_only` or `none` | Challenge dataset and NOAA HOMR | CC-BY-4.0 | Jumps in 1955–2014 | Matched within ±1 year | Analysis only |

## Not uploaded (download instead)

| Source | Exact version | How to download | Why not uploaded |
|---|---|---|---|
| Challenge dataset (`annual_indices`, `daily`, `city_info`) | Hugging Face commit `bbbdebdab2a18e00f97ef87bd3f35a71e222180b` | Loaded automatically by the notebook with `load_dataset(..., revision=REVISION)` | Already public and versioned on Hugging Face |
| Submission template (row order) | Hugging Face commit `09e2cefb5e6b64b4fba65d6c3e74c47277d7c0a4` | Loaded automatically by the notebook | Already public and versioned on Hugging Face |

## How the files fit together

- **Join key:** `ID` / `city_id` is the station `ID` in the challenge's `city_info` table, which is also the NOAA GHCN-Daily station ID. The backtest file uses `city`, the position in the sorted list of IDs.
- **Units:** °F for temperature, inches for precipitation and °F-days for CDD, unchanged from the source. `target` uses the dataset's names (`Avg Temp`, `ToT Prcp`, `Max Tmax`, `cdd`), and `metric` uses the submission's codes.
- **Known limitations:**
  - Infilled and corrected values are estimates.
  - Some cities have many years failing the 10-missing-day rule, such as Huntsville (29 of 30 years in 1989–2018).
  - NOAA station histories cover only each city's current station.
  - The NOAA files are a snapshot, so later NOAA updates may differ.

## Data rule check

- [x] Every file marked "Forecast input" is weather data up to 2018-12-31, or data that does not respond to the weather.
- [x] Files with 2019–2025 data that responds to the weather are marked "Analysis only" and are not read by the forecasting code. No such files are included. The NOAA files are station metadata with events up to 2018 only, and the forecasting code doesn't read them.
