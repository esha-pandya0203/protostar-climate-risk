# Curated data

Every file here is written by `code/Protostar_Final_Pipeline.ipynb`, and rerunning it regenerates them.

- **Source:** the only weather data is [`CMU-OSPO/protostar-climate-train-to-2018`](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018), with records up to 2018-12-31. No other source was used.
- **Terms:** the files are derived from that dataset and follow its terms.
- **Units:** °F for temperature, inches for precipitation and °F-days for CDD.
- **Identifiers:** `ID` and `city_id` are the dataset's station identifiers.
- **Metric names:** `target` uses the dataset's names (`Avg Temp`, `ToT Prcp`, `Max Tmax`, `cdd`). `metric` uses the submission's codes (`annual_avg_temp`, `total_precip_in`, `hottest_day_temp`, `cdd`).

## Values used by the forecasters

| File | Contents |
|---|---|
| `labels_final.csv.gz` | One row per city, year and target. `official` is the dataset value. `label_before_homogenisation` is the value after filling missing days (Avg Temp and CDD only). `homogenisation_adjustment` is the station-change correction applied at 2018. `label_used_at_2018` is the value the forecasters read. `source` says whether the value is official or infilled, and `missing_days` counts missing days. |
| `level_shifts_at_2018.csv` | Station-change jumps detected for Avg Temp and CDD against the median of each city's 8 nearest neighbours: `break_year`, `shift` (added to earlier years) and the t-statistic `t` |

## Quality audit (recorded, not removed)

| File | Contents |
|---|---|
| `quality_flags.csv` | One row per city-year. It gives missing days for daily high, daily low, precipitation and high/low pairs, and how much infilling changed CDD and Avg Temp. It also shows whether each target passes the official 10-missing-day rule (`strict_valid_*`), and the days in suspicious stuck-high runs. |
| `stuck_tmax_runs.csv` | Runs of at least 7 consecutive days with the same daily high, with how many such runs chance would produce (`expected_runs`). `suspicious` marks runs that are very unlikely by chance. |

## Model selection and forecasts

| File | Contents |
|---|---|
| `final_selection_tables.csv` | Backtest error for each candidate rule, start years 1990–1999 and 2000–2011. The rule with the lowest era-balanced error is selected. |
| `final_selected_methods.csv` | The rule selected for each target |
| `submission_team_only.csv` | The team's rules alone, in the same format as `submission.csv`, kept for comparison |
| `ensemble_components_2019_2025.csv` | For every submission row: `team`, `ridge`, `factor` and the submitted `ensemble` (0.50 / 0.25 / 0.25). An empty `ridge` or `factor` means that model had no forecast, and the team value was used in its place. |
| `ensemble_backtest_1990_2011.csv.gz` | The backtest rows behind the ensemble's score: start year (`origin`), target, city index (the position in the sorted list of station IDs), year, actual value, the 30-year baseline (`clim30`) and each model's forecast |
