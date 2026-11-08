# Protostar Climate Challenge: Where We Stand

## The challenge

Pretend it is the end of 2018. For 210 US cities, predict four yearly numbers for 2019–2025:

- **Average temperature**
- **Total precipitation** (rain plus melted snow)
- **Hottest day of the year**
- **Cooling degree days (CDD):** a measure of how much cooling a year's heat calls for

**The key rule:** we may not use weather from after 2018, or tune our models against it.

**How forecasts are scored:** we are compared with a simple guess, each city's 1989–2018 average. A score of 0 means no better than that guess, and 100 means perfect.

## What the data exploration (EDA) found

Details are in `eda.ipynb` and `eda_handoff_report.pdf`.

| Question | Finding | What it means |
|---|---|---|
| Is the data complete? | About 11–15% of city-years fail the rule of at most 10 missing days. Older years are worst. | Check quality separately for each of the four numbers. |
| Are extreme values errors? | Only about 1.2–1.4% of values were flagged as unusual, and some came from complete records. | Keep extremes unless there is evidence of an error. |
| Is it warming? | About 96% of cities show rising temperature and CDD over 1989–2018. | Recent years matter more than the full record. |
| Are all four numbers equally predictable? | Temperature and CDD follow clearer trends. Precipitation and hottest day are mostly noise from year to year. | Each number may need its own method. |
| Do nearby cities move together? | Often yes. Pittsburgh's temperature swings correlate 0.92–0.94 with nearby cities. | Neighbouring cities can help estimate trends. |
| Does this year predict next year? | Only weakly, especially for precipitation. | Past values alone give limited help seven years out. |
| Is the data stable over time? | The EDA's ADF test suggested most cities are, once a trend is allowed for. | Complex time-series models may add little. |
| Are the weather records consistent? | 180 of 210 cities combine records from more than one station. 19 station changes looked suspicious. | Some jumps in the data may come from station moves, not weather. |

## How the team's forecast works

`Protostar_Final_Pipeline.ipynb`, cleans the data and then applies simple rules. It does not use machine learning.

**Cleaning:** it estimates missing days for temperature and CDD, and corrects jumps caused by station changes. Its checks confirm the data's quality counts and that no future data is used.

| Number | Forecast rule |
|---|---|
| Average temperature | Average of the last 10 years, plus part of the recent warming trend |
| CDD | Recent daily temperatures, warmed by the expected amount, then CDD recalculated |
| Hottest day | Middle value (median) of the last 30 years |
| Precipitation | Middle value (median) of the last 30 years |

## Currently

1. **Reran the notebook from scratch.** It ran without errors, matched its saved model-selection tables and produced all 5,880 forecasts required.
2. **Added the missing comparison.** The notebook never tested itself against the 30-year average, which is what the official score uses, so we added that.
3. **Tested other models** on past data, including machine learning and ARIMA, a classic forecasting method.

We tested the way a real forecast works: stand at a past year, use only earlier data, and predict the next seven years. We repeated this for 1990 through 2011, so each model was tested 22 times. 

### Results

Higher is better. These are scores on past years, not predictions of the real result.

| Model | Score |
|---|---|
| Team's rules + regional model, averaged | 4.78 |
| Team's rules + Ridge regression (simple machine learning), averaged | 4.68 |
| **Team's current rules** | **4.35** |
| Regional model alone | 3.32 |
| Ridge regression alone | 3.27 |
| Best ARIMA | 2.13 |
| Automatically chosen ARIMA | 1.42 |
| Repeating last year's value | −26.58 |

Random forest (1.57) and gradient boosting (0.83) scored below the team's rules in an earlier run on slightly different rows.

### What this tells us

- **Keep the team's rules.** Every model tested on its own scored lower.
- **The two blends are not a clear win.** Their small lead is within normal variation and could be luck, since they are the best of many attempts.
- **More complex models did not help.** Long-range yearly weather is too noisy for them to find much more signal.
- **Accuracy comes mainly from temperature and CDD.** For hottest day and precipitation, even the team's rules only match the 30-year average.
- **Forecasts tend to run too cold in recent years.** 

### Was the data prepared properly?

Yes, for the methods the team uses. Missing data, outliers and station changes were all handled. The EDA tested stationarity (whether a series stays stable around a level or trend) once, using the full history.

ARIMA needs that decision made fresh at each point in time, so we added two tests (ADF and KPSS) inside every run. They found that only about 1–3% of cities need differencing, a transformation for unstable series, which fits the team's simpler approach.

## What's needed next

**Required to submit:**

- [ ] Decide to keep the team's rules, **before the first submission**. Choosing a model after seeing the Live Score breaks the rules.
- [ ] Create a public Hugging Face dataset from the challenge template.
- [ ] Add `submission.csv`, the code and the data.
- [ ] Write the README: method, policy idea, feedback on the challenge's data and tools, and how to reproduce the results. Also list the AI tools used, including Claude Code for this work.
- [ ] Fill in the data description, `data/README.md`.
- [ ] Run the template's `check_submission.py`. We have not run it yet; our own equivalent checks passed.
- [ ] Submit the link while signed in as a registered team member.

**Strongly recommended:**

- [ ] Fix the software and dataset versions so anyone gets the same result.
- [ ] Turn the notebook into a reusable script.
- [ ] Add the earlier experiments (1–5b), which the report cites but are not in the repo.
- [ ] Add the 2011–2018 practice run that the template asks for.

## Files in this folder

| File | What it does | Run time |
|---|---|---|
| `model_comparison.py` | Compares the team's rules with baseline, smoothing and machine learning models | About 7.5 min |
| `arima_comparison.py` | Compares them with ARIMA and related models, and tests stationarity | About 23 min |
| `results/` | Saved scores from both scripts | n/a |

**To run:** from the repository root, install the packages listed at the top of each script, then run `python experiments/model_comparison.py` or `python experiments/arima_comparison.py`. Add `--quick` for a short test run.

Everything here uses data through 2018 only. These scripts do not change `submission.csv`.
