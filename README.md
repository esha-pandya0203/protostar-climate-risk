---
pretty_name: Protostar Climate Risk Intelligence — 2019–2025 forecasts
---
# Protostar Climate Risk Intelligence: 2019–2025 forecasts for 210 U.S. cities

**Negative result: no**

**Team:** Abdullah Arshad, Tanisha, Imama, Esha

We forecast four yearly measures for 210 cities over 2019–2025: average temperature, total precipitation, hottest day and cooling degree days (CDD). We used only records up to 2018-12-31.

- **What we submit:** an ensemble. Each forecast is a fixed blend of three components: the team's per-target rules (50%), a Ridge regression pooled over all cities (25%) and a regional factor model (25%).
- **How we chose it:** we tested it on past data, standing at each year from 1990 to 2011 and forecasting the next seven years. The ensemble scored **4.96**, against **4.30** for its strongest component alone (the team's rules) and 0 for the 30-year average. On this score, 0 means no better than the 30-year average and 100 means a perfect forecast.
- **Policy idea ([`POLICY_IDEA.md`](POLICY_IDEA.md)):** plan cooling capacity using forecasts like these rather than 30-year "normals", which in our tests consistently underestimated cooling demand.

## Repository contents

| Path                                    | Contents                                                                                                                                                                        |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `submission.csv`                      | 5,880 forecasts (210 cities × 7 years × 4 metrics) in the template's row order                                                                                                |
| `code/Protostar_Final_Pipeline.ipynb` | The full pipeline: loads the data, cleans it, tests the methods on past data, selects them and writes `submission.csv` and `data/`. It is saved with its outputs.           |
| `code/experiments/`                   | Scripts that compare other models with the team's rules on past data, and their results (`results/`)                                                                          |
| `code/requirements.txt`               | The exact package versions used to produce `submission.csv`                                                                                                                   |
| `data/`                               | Curated data written by the notebook: cleaned values, quality flags, detected station-change jumps, the forecast from each model and the backtest rows. See `data/README.md`. |

## Method

### Data and cleaning

The only weather data is the challenge dataset, [`CMU-OSPO/protostar-climate-train-to-2018`](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018): `annual_indices`, `daily` and `city_info`. The only other input is the official submission template, used for row order. The notebook stops if any input is dated after 2018.

- **Quality counts:** missing days are counted from `daily`. The notebook checks that these counts match the dataset's own `Miss` columns, and that the number of valid city-years under the official rule (at most 10 missing days) matches the expected totals.
- **Missing days (Avg Temp and CDD only):** a city-year may have up to 30 missing days. Each missing day is filled with that city's average for the same calendar day over the previous 10 years, smoothed over ±7 days.
- **Station changes (Avg Temp and CDD):** each city is compared with the median of its 8 nearest neighbours. A lasting step in that difference is treated as a station change, and earlier years are shifted to the current station's level. A step counts if its t-statistic is at least 5 and it is at least 0.6 °F or 120 °F-days, with at least 5 years on each side.
- **Precipitation and hottest day:** only official values that meet the 10-missing-day rule are used. In the team's earlier experiments, every adjustment we tried for these was neutral or worse.
- **Unusual values are kept, not removed:** spikes, implausible values and suspicious runs of identical daily highs are recorded in `data/quality_flags.csv`, because removing them never improved the forecasts.

### The submitted model: an ensemble

The submitted forecast is one model, an ensemble, that blends three components:

```
forecast = 0.50 × team's rules + 0.25 × Ridge + 0.25 × regional factor model
```

**Why blend them:**
- Each component uses different information. The team's rules look at each city on its own, Ridge learns patterns that repeat across all cities, and the regional model uses climate swings shared by many cities.
- Their errors are only partly correlated, so averaging cancels some of them, and the error drops for all four targets. It doesn't remove the cold bias: for CDD the ensemble runs slightly colder than the team's rules, −22 against −12 °F-days averaged over start years 1990–2011.
- On their own, Ridge (3.27) and the regional model (3.32) score below the team's rules (4.30). Blended, they lift the score to 4.96.

#### Component 1: the team's rules (50%)

| Target              | Team's rule, selected by testing on past data                                                                                                  |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Average temperature | `lag_mean10_a0.5`: mean of the last 10 years, plus half the neighbour-pooled 30-year trend projected forward                                 |
| CDD                 | `delta_a0.75`: the last 15 years of daily temperatures, each warmed by 0.75 × the expected temperature trend, with CDD recomputed from them |
| Hottest day         | `recent30_median`: median of the last 30 years                                                                                               |
| Precipitation       | `recent30_median`: median of the last 30 years                                                                                               |

For each target, the rule with the lowest era-balanced error was chosen. Era-balanced error is the average of the mean error over start years 1990–1999 and over 2000–2011, so the slower warming of the 1990s and the faster warming of the 2000s count equally. This selection rule was declared before the backtest was run. Cities with fewer than 5 values in the last 15 years use their nearest neighbours' forecasts plus their historical offset, except for precipitation.

#### Component 2: Ridge regression (25%)

One model per target, pooled over all cities and trained on data from 1950 on. It predicts the gap from the 30-year average using recent averages and medians, the 30-year spread and trend, the neighbours' average trend, how recent the data is, national and neighbour gaps, latitude, longitude and how many years ahead it is forecasting.

#### Component 3: regional factor model (25%)

The three main climate patterns shared across cities over the last 40 years, each projected forward as a trend plus a fading carry-over from recent years.

#### How the components are combined

- **The weights are fixed, not fitted.** They were set from the backtests in `code/experiments/ensemble_comparison.py`, without using any 2019+ data.
- **Missing forecasts:** if Ridge or the regional model has no forecast for a city, the team's forecast takes its place in the blend. That happens for 1–3 cities per metric for Ridge and 4–8 for the regional model.
- **No negative values:** precipitation and CDD are never allowed below 0.
- **Every part is published:** `data/ensemble_components_2019_2025.csv` gives each component's forecast and the ensemble value for every submission row.

### How we tested it, without using later data

- **Forecasting from past years:** at each start year, models use only data dated on or before it and forecast the next seven years. They are scored against the 30-year average up to that year, using the challenge's formula and only city-years that meet the 10-missing-day rule. We never use random cross-validation, because it would train on years after the ones being predicted.
- **Leakage tests:** scrambling every value after a start year must change no forecast. The test is run for the team's rules (Section 6 of the notebook) and for the Ridge and regional models (Section 8b).
- **Same rows for every model:** all models are scored on the same rows, and uncertainty is estimated by resampling calendar years, because errors are correlated across cities within a year.

### Results on past data (not the 2019–2025 result)

| Model                                            | Start years 1990–2011 | 1990–1999 | 2000–2011 |
| ------------------------------------------------ | ---------------------: | ---------: | ---------: |
| **Ensemble, 50/25/25 (submitted)**               |         **4.96** |       3.06 |       6.26 |
| Alternative blend: team 75% + regional 25%       |                   4.73 |       2.88 |       5.99 |
| Alternative blend: team 75% + Ridge 25%          |                   4.67 |       2.83 |       5.92 |
| Component alone: team's rules                    |                   4.30 |       2.46 |       5.53 |
| Component alone: regional model (separate run)   |                   3.32 |            |            |
| Component alone: Ridge (separate run)            |                   3.27 |            |            |
| Other model: best ARIMA (separate run)           |                   2.13 |            |            |
| Other models: random forest / gradient boosting (separate run) |  1.57 / 0.83 |            |            |
| Baseline: repeating last year's value (separate run) |             −26.6 |            |            |

The separate runs score slightly different rows, so compare them only roughly with the first four. Every component scores lower alone than in the ensemble.

- **The gain is modest.** The ensemble beats the team's rules by +0.69 points. The 95% bootstrap range is about 0.0 to +1.5, and the ensemble wins in about 97–98% of resamples.
- **It holds up without hindsight.** The ensemble beat the team's rules in 18 of 22 start years. If, at each start year from 2000, we pick among the four blends using only forecasts whose seven years had fully passed, the score is 6.16, against 6.26 for the ensemble. So little of the gain comes from having chosen it with hindsight.
- **Almost all the skill is in average temperature and CDD:** 11.2 and 8.2 points respectively. For hottest day and precipitation, no method beat the 30-year average by more than about 0.5 points.
- **Expect a wide range for any single seven-year window:** from about −2.4 to 8.1 across start years, with a median of 6.9 for start years 2000–2011. A run of unusual weather can push any method to near 0.

**Methods we tested and rejected:**

- any single component alone (the team's rules, Ridge or the regional model);
- random forest and gradient boosting;
- ARIMA: 8 fixed orders plus automatic order selection, with stationarity tested fresh at each start year using ADF and KPSS tests;
- Theta, exponential smoothing and trend lines;
- three corrections for the team's cold bias (best: 3.81, against 4.30).

Scripts and results for all of these are in `code/experiments/`.

**Limitation:** the forecasts still run slightly cold. In the backtest, CDD averages −58 °F-days for start years 2000–2011.

## Reproducing the forecasts

```bash
pip install -r code/requirements.txt
jupyter nbconvert --to notebook --execute --inplace code/Protostar_Final_Pipeline.ipynb
```

- **Output:** `submission.csv` and `data/` at the repository root, whether the notebook is run from the root or from `code/`.
- **Runtime:** a few minutes on a laptop, plus the dataset download from Hugging Face.
- **Tested with:** Python 3.14.6.
- **Check:** a fresh run from this layout reproduced `submission.csv` byte for byte.

The comparison scripts load the notebook's pipeline themselves, so nothing has to be run first:

```bash
python code/experiments/ensemble_comparison.py        # ensemble backtest (~1 min); --quick for a smoke test
python code/experiments/model_comparison.py           # baselines, smoothing, ML (~8 min)
python code/experiments/arima_comparison.py           # ARIMA, Theta, regional model, stationarity (~23 min)
python code/experiments/error_analysis.py --diagnostics-only
python code/experiments/bias_correction.py
```

## Reuse and extensibility

- **`data/labels_final.csv.gz`** gives, for each city-year and metric, the official value, the infilled value, the station-change adjustment and the value used at 2018. Other teams can start from cleaned values instead of rebuilding the cleaning.
- **`data/level_shifts_at_2018.csv` and `data/quality_flags.csv`** are a city-by-city audit of station-change jumps, missing days and stuck daily highs in the source data (see Feedback to the Ecosystem).
- **The comparison scripts** run against the notebook itself rather than copies of it. A new model can be added to `code/experiments/` and scored on exactly the same rows as ours.
- **`data/ensemble_components_2019_2025.csv`** gives each model's forecast for every row, so other weightings can be tried without rerunning anything.

## Policy Idea

**Plan cooling for the climate we forecast, not the one we had.** The full policy idea (risk, action and who takes it, and why it is sustainable) is in [`POLICY_IDEA.md`](POLICY_IDEA.md).

## Feedback to the Ecosystem

**What worked:** the `Miss` columns in `annual_indices` let us recompute and reconcile every quality count exactly from `daily`, and the fixed cutoff made leakage easy to test.

**What got in the way, and what would fix it:**

1. **Station metadata misses most jumps in the records.** Comparing each city with its 8 nearest neighbours found 95 lasting jumps in average temperature, in 90 of the 210 cities. Only 13 fall within a year of a station change listed in `city_info`; for CDD it is 9 of 57. Examples:

   - Reno, +4.1 °F in 1999, with no change listed;
   - Phoenix, +2.5 °F (+732 CDD) in 1972, with no change listed;
   - Marquette, −3.6 °F in 1979, which does match the move to Marquette WFO.

   Some of these jumps may be real local change, such as urban growth. Without metadata a user cannot tell.
   **Fix:** publish instrument and siting changes, not only station-ID changes, and a list of detected breaks with their likely causes.
2. **Some cities' records are too gappy to use, and the dataset card doesn't say which.** In 1989–2018, Huntsville fails the 10-missing-day rule in 29 of 30 years for every metric. Elko and Grand Island fail it in 29 of 30 years for precipitation only. Belvidere (27), Escondido (22) and Riverside (18) also fail it in many years.
   **Fix:** add a per-city, per-metric coverage table to the dataset card. The `coverage_and_stations` subset added on November 5 goes part of the way. We did not use it, because our pipeline computes coverage from `daily`.
3. **Daily values have no quality-control flags.** We found 43 runs where the daily high repeated the same value for a week or more, which is statistically unlikely. They include recent data, such as Galveston at 91 °F for 11 days from 2016-07-31.
   **Fix:** carry the source dataset's quality flags into `daily`.
4. **The scoring baseline needs a reference implementation.** The rule that a baseline needs at least 10 valid years was clarified on November 6, after teams had computed baselines without it.
   **Fix:** publish a small scoring script that computes the baselines and skill, including the 2011–2018 practice run.

## AI tools used

- _TODO: list the AI tools the team used._

No AI model was used as a forecasting input, so no model trained on post-2018 data feeds the forecasts.

## External data sources

None. The only inputs are `CMU-OSPO/protostar-climate-train-to-2018` and the row order from `CMU-OSPO/protostar-climate-submission-template`.
