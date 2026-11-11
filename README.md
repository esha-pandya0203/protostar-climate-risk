---
pretty_name: Protostar Climate Risk Intelligence — 2019–2025 forecasts
---
# Protostar Climate Risk Intelligence: 2019–2025 forecasts for 210 U.S. cities

**Negative result: no**

**Team:** Abdullah Arshad, Tanisha, Imama, Esha

We forecast four yearly measures for 210 cities over 2019–2025: average temperature, total precipitation, hottest day and cooling degree days (CDD). We used only records up to 2018-12-31.

- **What we submit:** an ensemble. Each forecast is a fixed blend of three components: the team's per-target rules (50%), a Ridge regression pooled over all cities (25%) and a regional factor model (25%).
- **How we chose it:** we tested it on past data, standing at each year from 1990 to 2011 and forecasting the next seven years. The ensemble scored **4.96**, against **4.30** for its strongest component alone (the team's rules) and 0 for the 30-year average. On this score, 0 means no better than the 30-year average and 100 means a perfect forecast.
- **Policy idea:** a Climate-Adjusted Cooling Planning Standard. City buildings and utilities would plan cooling with warming-adjusted forecasts instead of 30-year "normals", which in our tests consistently underestimated cooling demand (also in [`POLICY_IDEA.md`](POLICY_IDEA.md)).

## Repository contents

| Path                                    | Contents                                                                                                                                                                        |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `submission.csv`                      | 5,880 forecasts (210 cities × 7 years × 4 metrics) in the template's row order                                                                                                |
| `code/Protostar_Final_Pipeline.ipynb` | The full pipeline: loads the data, cleans it, tests the methods on past data, selects them and writes `submission.csv` and `data/`. It is saved with its outputs.           |
| `code/experiments/`                   | Scripts that compare other models with the team's rules on past data, and their results (`results/`)                                                                          |
| `code/integration/`                   | Joins the challenge stations with NOAA GHCN-Daily and NOAA station histories (metadata only), and checks every detected station-change jump against them |
| `code/requirements.txt`               | The exact package versions used to produce `submission.csv`                                                                                                                   |
| `data/`                               | Curated data written by the notebook: cleaned values, quality flags, detected station-change jumps, the forecast from each model and the backtest rows. See `data/README.md`. |

## Method

### Data and cleaning

The only weather data is the challenge dataset, [`CMU-OSPO/protostar-climate-train-to-2018`](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018): `annual_indices`, `daily` and `city_info`. The only other input is the official submission template, used for row order. The notebook stops if any input is dated after 2018. Both Hugging Face datasets are loaded at pinned versions. NOAA station metadata is used only to audit the data (see Interoperability and integration), never as a forecast input.

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
- **Data versions:** the notebook loads the challenge dataset at commit `bbbdebdab2a18e00f97ef87bd3f35a71e222180b` and the template at `09e2cefb5e6b64b4fba65d6c3e74c47277d7c0a4`, so later updates to either cannot change the results.
- **Check:** a fresh run from this layout reproduced `submission.csv` byte for byte.

The comparison scripts load the notebook's pipeline themselves, so nothing has to be run first:

```bash
python code/experiments/ensemble_comparison.py        # ensemble backtest (~1 min); --quick for a smoke test
python code/experiments/model_comparison.py           # baselines, smoothing, ML (~8 min)
python code/experiments/arima_comparison.py           # ARIMA, Theta, regional model, stationarity (~23 min)
python code/experiments/error_analysis.py --diagnostics-only
python code/experiments/bias_correction.py
python code/integration/noaa_station_metadata.py    # NOAA metadata join and break audit (~1 min, needs internet)
```

## Reuse and extensibility

- **`data/labels_final.csv.gz`** gives, for each city-year and metric, the official value, the infilled value, the station-change adjustment and the value used at 2018. Other teams can start from cleaned values instead of rebuilding the cleaning.
- **`data/level_shifts_at_2018.csv` and `data/quality_flags.csv`** are a city-by-city audit of station-change jumps, missing days and stuck daily highs in the source data (see Feedback to the Ecosystem).
- **The comparison scripts** run against the notebook itself rather than copies of it. A new model can be added to `code/experiments/` and scored on exactly the same rows as ours.
- **`data/ensemble_components_2019_2025.csv`** gives each model's forecast for every row, so other weightings can be tried without rerunning anything.

## Interoperability and integration

The challenge's station IDs (for example `USW00023185`, Reno) are standard NOAA GHCN-Daily identifiers, and every file in `data/` is keyed by them. Our outputs therefore join directly with NOAA's records, and with any other dataset that uses GHCN-Daily IDs. `code/integration/noaa_station_metadata.py` makes that join in about a minute. NOAA station metadata isn't weather data, and only events dated up to 2018-12-31 are kept. It doesn't feed the forecasts.

- **NOAA GHCN-Daily station list:** all 210 IDs are found. The join adds elevation, and flags the 40 stations in NOAA's long-term climate networks (HCN/CRN). It also shows that for 18 stations, the challenge dataset's coordinates are more than 1 km from NOAA's: up to 11.1 km for Corpus Christi and 6.0 km for Los Angeles (`data/ghcnd_station_crosswalk.csv`).
- **NOAA station histories (HOMR):** 644 dated relocations and changes of temperature equipment up to 2018 for the 210 stations (`data/noaa_homr_events_to_2018.csv`). `city_info` lists only switches between named stations, so most of these events aren't in the challenge data.
- **Cross-check of our station-change correction:** our pipeline corrected 95 average-temperature jumps. Of these, 13 match a `city_info` change within a year, and 18 more match a NOAA relocation or equipment change. Counting only jumps in years that the current station's NOAA history covers, 29 of 76 are explained and **47 (62%) have no documented cause** (`data/station_break_attribution.csv`). Example: Los Angeles's −1.9 °F jump in 1999 matches a 3.5-mile relocation in July 1999 that `city_info` doesn't list.
- **What limited the integration:**
  - The join itself was easy because the IDs are shared.
  - NOAA's station history covers only a city's current station, so jumps from earlier stations in a combined record can't be checked against it.
  - NOAA's JSON service sometimes returned truncated responses under parallel requests, which the script handles with retries.
  - Coordinates disagree between the two sources, so joining on location instead of ID would mismatch some stations.

## Who would use this, and could it work in practice?

- **Utilities and public utility commissions:** the CDD forecasts and the policy idea, for peak-load and capacity planning. It's practical because it needs only public station data and runs in minutes. It has limits: the forecasts are yearly city totals, which a utility would still have to translate into peak loads, and their gain over 30-year normals is modest. Most of that gain is in temperature and CDD.
- **Researchers and future challenge teams:** cleaned values, audit files, and a test harness that scores any new model on exactly our rows.
- **Dataset maintainers (CMU-OSPO, and NOAA for the underlying records):** a list of 152 detected jumps, each matched against two metadata sources, as a starting point for documenting station history.

## What is new

- **CDD from warmed daily temperatures:** instead of extending a yearly CDD trend, we warm each of the last 15 years of daily temperatures by the projected trend and recompute CDD. Only days above 65 °F count towards CDD, and this approach accounts for that.
- **Cleaning that is tested for leakage too:** the station-change correction uses only data up to each start year, so the cleaning itself passes the same leakage test as the forecasts.
- **An ensemble of components that see different information** (one city, all cities, regional patterns), with fixed weights and a walk-forward check that the choice holds up without hindsight.
- **An automated audit of station-change jumps against two independent metadata sources**, turning a data-cleaning step into feedback the dataset maintainers can act on.

## Policy Idea: Climate-Adjusted Cooling Planning Standard

_Plan cooling for the climate we forecast, not the one we had._

**What risk did we notice?** Our backtests found that 30-year climate averages underestimated cooling degree days (CDD) by 118°F-days per city-year, with actual CDD exceeding historical normals in 70% of cases. Our 2019–2025 forecasts project higher cooling demand across all 207 evaluated U.S. cities, with a median increase of 11% over historical averages. We also identified undocumented weather station changes and inconsistent records that can distort climate projections, highlighting the need for more reliable, forward-looking climate planning.

**What would we do about it, and who would do it?** Cities should establish a Climate-Adjusted Cooling Planning Standard with local utilities and neighboring municipalities. New and substantially renovated city-owned buildings would be required to use warming-adjusted CDD forecasts for energy-efficient cooling design. Participating utilities would update these projections annually for electricity peak-load and resource planning, with mandatory compliance requiring utility regulation. A shared regional weather database would document station changes and use validated neighboring observations to address missing or inconsistent records, particularly benefiting smaller communities.

**Why is it sustainable?** Integrating forecasts into existing construction approvals and utility planning cycles keeps implementation manageable. Shared weather records reduce duplicated data-processing costs, while passive cooling and building efficiency limit electricity demand and emissions. Earlier utility planning supports demand-response and cleaner energy investments instead of expensive emergency power purchases or unnecessary fossil-fuel capacity. These measures help cities maintain reliable cooling without escalating long-term costs or environmental impacts.

The same text is available as a standalone document in [`POLICY_IDEA.md`](POLICY_IDEA.md).

## Feedback to the Ecosystem

**What worked:** the `Miss` columns in `annual_indices` let us recompute and reconcile every quality count exactly from `daily`, and the fixed cutoff made leakage easy to test.

**What got in the way, and what would fix it:**

1. **Station metadata misses most jumps in the records, even after adding NOAA's.** Comparing each city with its 8 nearest neighbours found 95 lasting jumps in average temperature, in 90 of the 210 cities.
   - Only 13 fall within a year of a station change listed in `city_info`.
   - NOAA's station histories explain 18 more. That still leaves 47 of the 76 jumps that NOAA's history can check (62%) with no documented cause.
   - For CDD, 9 of 57 jumps match `city_info` and 9 more match NOAA.

   Examples:
   - Reno, +4.1 °F in 1999: NOAA records a relocation in 1995 and a sensor change in 2001, but nothing within a year of the jump.
   - Phoenix, +2.5 °F (+732 CDD) in 1972: nothing in either source.
   - Los Angeles, −1.9 °F in 1999: matches a 3.5-mile relocation that only NOAA lists.

   Some of these jumps may be real local change, such as urban growth. Without metadata a user cannot tell.
   **Fix:** carry NOAA's relocation and equipment history into `city_info`, keyed by GHCN-Daily ID and including earlier stations. Publish detected breaks, such as our `data/station_break_attribution.csv`, with their likely causes.
2. **Some cities' records are too gappy to use, and the dataset card doesn't say which.** In 1989–2018, Huntsville fails the 10-missing-day rule in 29 of 30 years for every metric. Elko and Grand Island fail it in 29 of 30 years for precipitation only. Belvidere (27), Escondido (22) and Riverside (18) also fail it in many years.
   **Fix:** add a per-city, per-metric coverage table to the dataset card. The `coverage_and_stations` subset added on November 5 goes part of the way. We did not use it, because our pipeline computes coverage from `daily`.
3. **Daily values have no quality-control flags.** We found 43 runs where the daily high repeated the same value for a week or more, which is statistically unlikely. They include recent data, such as Galveston at 91 °F for 11 days from 2016-07-31.
   **Fix:** carry the source dataset's quality flags into `daily`.
4. **The scoring baseline needs a reference implementation.** The rule that a baseline needs at least 10 valid years was clarified on November 6, after teams had computed baselines without it.
   **Fix:** publish a small scoring script that computes the baselines and skill, including the 2011–2018 practice run.
5. **Station coordinates disagree with NOAA's.** For 18 of 210 stations, the coordinates in the challenge dataset are more than 1 km from NOAA's for the same ID: Corpus Christi by 11.1 km, Los Angeles by 6.0 km and Chicago by 3.9 km. Any distance-based method, such as nearest neighbours or regional pooling, inherits this error.
   **Fix:** publish coordinates for each station period, or state which station the coordinates refer to.

## AI tools used

- **Claude Code (Anthropic, model Claude Opus 5.5).** We used it to rerun and review the pipeline, write and run the comparison scripts in `code/experiments/`, add the ensemble to the notebook and check it against an independent recomputation, build the NOAA metadata audit in `code/integration/`, pull the policy and feedback examples from the data, and draft the documentation. The team checked its outputs against the saved results and reviewed all of the text.

No AI model was used as a forecasting input, so no model trained on post-2018 data feeds the forecasts.

## External data sources

- **Forecast inputs:** only [`CMU-OSPO/protostar-climate-train-to-2018`](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018), plus the row order from `CMU-OSPO/protostar-climate-submission-template`.
- **Audit only, not used by the forecasts:** NOAA station metadata, with events up to 2018-12-31 only.
  - [GHCN-Daily station list](https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt)
  - [NOAA Historical Observing Metadata Repository (HOMR)](https://www.ncei.noaa.gov/access/homr/), the station histories
