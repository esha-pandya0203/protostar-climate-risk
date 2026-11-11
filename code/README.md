# Code

- `Protostar_Final_Pipeline.ipynb`: the full pipeline. It loads the pinned training data, cleans it, tests the candidate methods on past data, builds the ensemble (team rules 50%, Ridge 25%, regional factor model 25%) and writes `submission.csv` and `data/`. Run it from top to bottom; it is saved with its outputs.
- `requirements.txt`: the exact package versions used to produce `submission.csv`.
- `experiments/`: backtests comparing the ensemble's components and other models (baselines, smoothing, machine learning, ARIMA, bias corrections). Each script loads the notebook's pipeline itself. Results are in `experiments/results/`.
- `integration/noaa_station_metadata.py`: joins the stations with NOAA GHCN-Daily and NOAA station histories (metadata only, events up to 2018), and checks the detected station-change jumps against them.
- `check_submission.py` (from the template): checks `submission.csv` before submitting. Run `python code/check_submission.py`.
- `AGENTS.md` (from the template): instructions for AI agents, including the data rule.
