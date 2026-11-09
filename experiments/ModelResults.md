# Protostar Climate Risk Intelligence — Historical Climate Forecasting

This repository contains our data exploration, forecasting pipeline, historical backtests, and model-comparison experiments for the Protostar Climate Risk Intelligence Challenge.

**Task:** using information available by **December 31, 2018**, forecast four annual climate measurements for **210 U.S. cities** over **2019–2025** (7 years × 4 metrics × 210 cities = **5,880 predictions**). The four targets are:

| Target | Description |
|---|---|
| `Avg Temp` | Annual average temperature (°F) |
| `ToT Prcp` | Total annual precipitation (inches) |
| `Max Tmax` | Highest daily maximum temperature in the year (°F) |
| `cdd` | Annual cooling degree days above a 65°F reference temperature (degree-days) |

**Information cutoff:** no observations dated after 2018 are used for modeling or tuning. Each historical backtest origin uses only information available by that origin. Historical scores below **are not the challenge's 2019–2025 results**.

## 1. Data exploration and preprocessing

The EDA examines three source tables: `annual_indices` (28,052 city-years), `daily` (10,245,754 observations), and `city_info` (461 station-period records). See `eda.ipynb` and `eda_handoff_report.pdf` for the analysis and visual summary.

| EDA finding | Modeling implication |
|---|---|
| Approximately 11–15% of city-years fail the initial target-specific ≤10 missing-day screen | Validate completeness for each target rather than assuming non-null annual values are trustworthy |
| Approximately 96% of cities have positive recent temperature and CDD trends | Test trend-aware methods for these targets |
| Temperature and CDD show clearer trends than precipitation or yearly hottest-day temperature | Choose forecasting rules by target |
| Nearby cities share annual temperature fluctuations | Explore regional information sharing |
| Station histories can introduce discontinuities | Investigate and adjust station-related level shifts where justified |
| A small share of values are flagged as unusual | Retain plausible weather extremes rather than dropping them automatically |

The core pipeline checks historical record completeness; applies climatological missing-temperature infilling and station homogenisation for **average temperature and CDD**; and retains the stricter completeness screen for **precipitation and maximum temperature**. Outliers are investigated but not automatically discarded. For the formal data-quality experiments and their limitations, see `Protostar_Final Report.pdf`.

## 2. Original team forecasting pipeline

`Protostar_Final_Pipeline.ipynb` is the original, runnable forecasting pipeline. Its selected methods are:

| Target | Method | Description |
|---|---|---|
| Average temperature | `lag_mean10_a0.5` | Recent 10-year mean, corrected using a fraction of the estimated warming trend |
| Total precipitation | `recent30_median` | Median of the last 30 years |
| Maximum temperature | `recent30_median` | Median of the last 30 years |
| CDD | `delta_a0.75` | Warm historical daily temperatures by an estimated trend adjustment, then recalculate CDD |

The notebook evaluates methods with chronological rolling-origin backtests, checks that future observations do not enter earlier forecasts, and generates the original model's **`submission.csv`** in its final-forecast section (Section 9) with:

```python
sub.to_csv("submission.csv", index=False)
```

**Important implementation status:** `ensemble_comparison.py` only evaluates alternative forecasts; it **does not** change the notebook or submission. Until the ensemble is explicitly integrated and the output regenerated, `submission.csv` from the original notebook is the **team model submission**, not an ensemble submission.

## 3. Historical evaluation design

The comparison scripts evaluate seven-year-ahead forecasts from rolling origins **1990–2011** (22 origins). The main scoring reference is each city's preceding 30-year climatological mean, evaluated on valid official annual observations. A model's target-level skill is

\[
\mathrm{Skill}=100\left(1-\frac{\mathrm{MAE}_{model}}{\mathrm{MAE}_{clim30}}\right).
\]

Skill is averaged across the four metrics for an overall comparison; **higher is better**, zero matches the reference and negative values indicate worse error. We also inspect mean absolute error (MAE), signed bias (`predicted - actual`), era-specific scores and calendar-year bootstrap uncertainty. Because historical origins and forecast windows overlap, and several candidates were compared, apparent improvements should be interpreted as **exploratory rather than independent final-test results**.

## 4. Model experiments and results

### Statistical, machine learning, and time-series comparisons

`model_comparison.py` compares simple statistical rules, exponential smoothing, robust trends, Ridge regression, Random Forest, and Gradient Boosting. `arima_comparison.py` extends this with ARIMA configurations, Theta, a three-factor shared-climate model, and origin-specific ADF/KPSS stationarity checks.

Across these earlier experiments, the original team approach was competitive; no individual alternative clearly surpassed it. Regional factor and Ridge forecasts were nevertheless useful inputs for ensemble tests. Scores from different scripts may differ slightly because they do not always evaluate exactly the same matched rows; compare candidates within the same experiment.

### Error analysis and historical bias correction

`error_analysis.py` breaks original-team errors down by **forecast horizon, city, and origin**. Average-temperature and CDD predictions exhibited increasing negative bias at longer horizons, motivating further tests.

`bias_correction.py` tested three *walk-forward* error corrections (using only completed past forecast blocks):

| Method | Historical overall skill, 1990–2011 |
|---|---:|
| Original team | **4.30** |
| Half-strength global bias correction | 3.81 |
| Full global bias correction | 2.09 |
| Neighbor-pooled regional bias correction | −0.30 |

These methods reduced underprediction in some recent periods but did **not** improve overall skill; they were not adopted.

### Fixed-weight ensemble comparison — latest corrected historical evaluation

`ensemble_comparison.py` combines three components: (1) original target-specific team forecasts, (2) pooled **Ridge regression** trained on historical city/region/geographic features, and (3) a **three-factor regional model** that extracts and forecasts shared climate patterns. The fixed candidate weights were:

| Candidate | Team | Ridge | Regional factor |
|---|---:|---:|---:|
| `team_selected` | 100% | 0% | 0% |
| `ridge_25` | 75% | 25% | 0% |
| `factor_25` | 75% | 0% | 25% |
| `combined_25_25` | 50% | 25% | 25% |

The corrected evaluation requires valid truth, team, and climatology predictions, and **falls back to the team prediction for a missing Ridge or factor component** rather than dropping the entire observation. All **four targets** and **22 historical origins** were retained, yielding **121,742 evaluated prediction rows**. Ridge coverage was 100%; regional-factor coverage was approximately 99.4%.

| Model | 1990–2011 skill | 1990–1999 | 2000–2011 | 2005–2011 |
|---|---:|---:|---:|---:|
| Original team | 4.30 | 2.46 | 5.53 | 5.13 |
| Team + 25% Ridge | 4.67 | 2.83 | 5.92 | 5.50 |
| Team + 25% regional factor | 4.73 | 2.88 | 5.99 | 5.69 |
| **Team + 25% Ridge + 25% regional factor** | **4.96** | **3.06** | **6.26** | **5.98** |

**MAE of the original team versus the combined ensemble** (matched historical sample):

| Target | Original team MAE | Combined ensemble MAE |
|---|---:|---:|
| Average temperature | 1.151°F | **1.139°F** |
| Maximum temperature | 2.643°F | **2.628°F** |
| Total precipitation | 5.771 in | **5.754 in** |
| CDD | 162.015 | **160.546** |

The combined ensemble's estimated skill difference against the team was **+0.689 points**, with a calendar-year bootstrap 95% interval of **+0.031 to +1.489** and `P(beats team) = 0.9805`. These are promising *historical-development* findings, not a guarantee of improved challenge performance. The selected best candidate has been chosen after comparing multiple methods on overlapping historical folds.

**Model decision:** The **50/25/25 combined ensemble is the strongest candidate from this corrected comparison**. It should only be described as the **final submitted model** once its 2018-origin forecasts, fallback logic, validation, and generated `submission.csv` have been integrated and verified. Otherwise, the original team pipeline remains the implemented submission method.

## 5. Repository guide

The following table describes the **recommended files to commit**. Keep one project README at the root; the prior standalone `error_analysis.md` is merged into this document.

| File | Contents |
|---|---|
| `README.md` | Project overview, methods, results, usage, and repository file guide (this file) |
| `eda.ipynb` | Exploratory analysis of data quality, missingness, trends, outliers, geography, and station history |
| `eda_handoff_report.pdf` | Visual EDA findings and modeling implications shared with the forecasting team |
| `Protostar_Final_Pipeline.ipynb` | Primary cleaning, historical backtest, model selection, and original submission generation |
| `Protostar_Final Report.pdf` | Data-quality audit and historical forecasting experiments informing the final pipeline |
| `experiments/model_comparison.py` | Baseline/statistical/ML forecasts and chronological historical comparisons |
| `experiments/arima_comparison.py` | ARIMA/Theta/regional-factor comparison and ADF/KPSS stationarity diagnostics |
| `experiments/error_analysis.py` | Breaks down original-team errors by horizon, city and origin; also contains experimental trend/neighbor candidate routines |
| `experiments/bias_correction.py` | Tests global, half-strength and neighborhood-pooled walk-forward bias corrections |
| `experiments/ensemble_comparison.py` | Evaluates the four fixed-weight ensemble candidates and reports component coverage; **commit the corrected version** |
| `experiments/results/scores_by_era.csv` | Overall scores of baseline/statistical/ML models across historical eras |
| `experiments/results/metrics_1990_2011.csv` | Baseline/statistical/ML MAE, bias and skill by target |
| `experiments/results/bootstrap_vs_team.csv` | Calendar-year bootstrap comparisons against the team forecast |
| `experiments/results/walk_forward_picks.csv` | Models selected at successive historical origins using earlier information |
| `experiments/results/arima_scores_by_era.csv` | Scores of the ARIMA/Theta/factor candidate set by era |
| `experiments/results/arima_metrics_1990_2011.csv` | ARIMA/Theta/factor MAE, bias and skill by target |
| `experiments/results/arima_bootstrap_vs_team.csv` | Bootstrap comparisons of time-series alternatives with team forecasts |
| `experiments/results/stationarity_at_2018.csv` | ADF/KPSS stationarity classifications by climate target at the 2018 origin |
| `experiments/results/bias_correction_scores.csv` | Historical skill for the three bias corrections and team model |
| `experiments/results/bias_correction_bootstrap.csv` | Bootstrap uncertainty for bias-correction experiments |
| `experiments/results/ensembles/ensemble_scores_by_era.csv` | **Corrected** four-target ensemble scores over historical periods |
| `experiments/results/ensembles/ensemble_metrics_by_target.csv` | **Corrected** ensemble MAE, bias and skill for each target |
| `experiments/results/ensembles/ensemble_metrics_by_target_era.csv` | **Corrected** target-level ensemble scores in each historical period |
| `experiments/results/ensembles/ensemble_bootstrap_vs_team.csv` | **Corrected** bootstrap skill differences relative to original team |
| `experiments/results/ensembles/ensemble_component_coverage.csv` | Ridge and factor prediction availability by target |
| `experiments/results/ensembles/ensemble_evaluation_coverage.csv` | Number of evaluated observations at each origin for each target |
| `experiments/results/error_analysis_results/error_by_horizon.csv` | Forecast MAE and signed bias by target and 1–7-year horizon |
| `experiments/results/error_analysis_results/error_by_origin.csv` | Forecast MAE and signed bias by target and historical origin |
| `experiments/results/error_analysis_results/error_by_city.csv` | City-level forecast MAE and signed bias, including identifiers |
| `experiments/results/error_analysis_results/horizon_*.png` | Four horizon-specific error charts, one for each climate target |
| `submission.csv` | 5,880 challenge predictions, **only after verifying which pipeline generated them** |

The three error-analysis CSVs and four horizon charts are useful supporting results; if minimizing repository size, keeping just `error_by_horizon.csv` and the charts is reasonable. Do not commit duplicate or superseded ensemble CSVs from the earlier, incomplete temperature-only run.

**Additional experimental files, if produced:** `paired_forecasts.csv`, `paired_candidate_metrics.csv`, and `paired_candidate_bootstrap.csv` are outputs from `error_analysis.py` for matched candidate forecast/error comparisons. They may be omitted from GitHub if they are large or incomplete; the generation code remains available. If committed, store them under `experiments/results/error_analysis_results/` and keep their provenance clear.

## 6. Reproduce the comparisons

From the repository root, install the Python dependencies:

```bash
python -m pip install datasets huggingface_hub pandas numpy matplotlib scikit-learn statsmodels joblib
```

Run the baseline notebook before or as part of the experiment scripts (these scripts load and execute its pre-selection cells):

```bash
python experiments/model_comparison.py
python experiments/arima_comparison.py
python experiments/error_analysis.py --diagnostics-only
python experiments/bias_correction.py
python experiments/ensemble_comparison.py
```

For short checks, `model_comparison.py`, `arima_comparison.py`, `error_analysis.py`, and `ensemble_comparison.py` support `--quick`. The comparison scripts can be computationally expensive because they rerun the notebook to load its data and original forecasts. These experiments **do not automatically regenerate the final submission**.