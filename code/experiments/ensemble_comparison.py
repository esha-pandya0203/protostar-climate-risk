"""Four fixed Protostar ensembles with explicit evaluation coverage.

Run at repo root:
    python code/experiments/ensemble_comparison.py --quick
    python code/experiments/ensemble_comparison.py

Place next to model_comparison.py and arima_comparison.py.
No 2019+ observations are used. Does not update submission.csv.

When a Ridge/factor forecast is unavailable for a city/horizon, use the
team forecast in its place *inside that ensemble only*. This keeps the
comparison paired and makes unavailable-component coverage auditable.
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import model_comparison as mc
from arima_comparison import factor_model

HORIZON = 7
FIRST_ORIGIN = 1990
KEY = ["origin", "target", "city", "year"]
WEIGHTS = {
    "team_selected": (1.00, 0.00, 0.00),
    "ridge_25": (0.75, 0.25, 0.00),
    "factor_25": (0.75, 0.00, 0.25),
    "combined_25_25": (0.50, 0.25, 0.25),
}
RESULTS = os.path.join(mc.RESULTS, "ensembles")


def collect_predictions(ns, origins):
    frames = []
    for target in ns["TARGETS"]:
        X = np.where(ns["STRICT"][target], ns["A"][target], np.nan)
        cache = mc.FeatureCache(X, ns)
        for origin in origins:
            idx = ns["yidx"](origin)
            truth = X[:, idx + 1:idx + 1 + HORIZON]
            if truth.shape[1] != HORIZON:
                raise ValueError(f"Incomplete 7-year truth window at origin {origin}")
            ridge = mc.ml_models(X, origin, cache, names=["ridge"])["ridge"]
            factor = factor_model(X, origin, ns["yidx"], k=3)
            clim = mc.statistical_models(X, origin, ns["years"], ns["yidx"])["clim30"]
            if target in ns["NONNEG"]:
                ridge, factor = np.maximum(ridge, 0), np.maximum(factor, 0)

            city, step = np.nonzero(np.isfinite(truth))
            frames.append(pd.DataFrame({
                "origin": origin, "target": target, "city": city,
                "year": origin + step + 1, "actual": truth[city, step],
                "clim30": clim[city, step],
                "ridge": ridge[city, step], "factor_3": factor[city, step]
            }))
        print(f"{target}: processed {len(origins)} origins", flush=True)

    candidates = pd.concat(frames, ignore_index=True)
    print("\nInitial valid truth rows by target:")
    print(candidates.groupby("target").size().to_string())

    bt, selection = ns["bt"], ns["selection"]
    team = bt.loc[bt.method == bt.target.map(selection), KEY + ["predicted"]].rename(
        columns={"predicted": "team_selected"})
    # Keep left side so we can inspect rows without matching team forecasts.
    candidates = candidates.merge(team, on=KEY, how="left", validate="one_to_one", indicator=True)
    print("\nTeam join status by target:")
    print(pd.crosstab(candidates.target, candidates["_merge"]).to_string())
    candidates = candidates.drop(columns="_merge")

    for col in ["actual", "clim30", "team_selected", "ridge", "factor_3"]:
        candidates[col] = candidates[col].replace([np.inf, -np.inf], np.nan)

    print("\nMissing predictions before filtering (by target):")
    print(candidates.groupby("target")[["clim30", "team_selected", "ridge", "factor_3"]]
          .agg(lambda v: v.isna().sum()).to_string())

    # Score only rows with actual truth, 30-year baseline, and TEAM forecast.
    # Ridge/factor are optional; fallback to team keeps all ensembles on identical rows.
    valid = candidates.dropna(subset=["actual", "clim30", "team_selected"]).copy()
    print(f"\nPaired team+clim rows: {len(valid):,} / {len(candidates):,}")
    print("Rows retained by target:")
    print(valid.groupby("target").size().to_string())
    print("Origin ranges retained by target:")
    print(valid.groupby("target").origin.agg(["min", "max", "nunique"]).to_string())

    availability = valid.groupby("target").agg(
        rows=("actual", "size"),
        ridge_available=("ridge", "count"),
        factor_available=("factor_3", "count"),
    )
    availability["ridge_pct"] = 100 * availability.ridge_available / availability.rows
    availability["factor_pct"] = 100 * availability.factor_available / availability.rows
    print("\nComponent availability (% among scored rows):")
    print(availability.round(2).to_string())

    missing_targets = set(ns["TARGETS"]) - set(valid.target.unique())
    if missing_targets:
        raise RuntimeError(f"Missing targets after mandatory filtering: {sorted(missing_targets)}")
    if len(origins) > 2:
        for target in ns["TARGETS"]:
            got = set(valid.loc[valid.target == target, "origin"].unique())
            missing_origins = set(origins) - got
            if missing_origins:
                raise RuntimeError(f"{target}: missing origins {sorted(missing_origins)}; check team/baseline coverage")

    ridge = valid.ridge.fillna(valid.team_selected)
    factor = valid.factor_3.fillna(valid.team_selected)
    for name, (a, b, c) in WEIGHTS.items():
        valid[name] = a * valid.team_selected + b * ridge + c * factor
        if name != "team_selected":
            nonneg = valid.target.isin(ns["NONNEG"])
            valid.loc[nonneg, name] = valid.loc[nonneg, name].clip(lower=0)

    if valid[list(WEIGHTS)].isna().any().any():
        raise RuntimeError("Unexpected NaN in final ensemble predictions")
    return valid, availability


def evaluate(wide, availability, quick=False):
    models = list(WEIGHTS)
    scores = mc.era_scores(wide, models)
    details = mc.score_table(wide, models).reset_index()
    era_rows = []
    for label, mask in [
        ("1990-1999", wide.origin.between(1990, 1999)),
        ("2000-2011", wide.origin.between(2000, 2011)),
        ("2005-2011", wide.origin.between(2005, 2011)),
    ]:
        if mask.any():
            part = mc.score_table(wide.loc[mask], models).reset_index()
            part.insert(0, "era", label)
            era_rows.append(part)
    era_detail = pd.concat(era_rows, ignore_index=True) if era_rows else pd.DataFrame()
    print("\n=== Skill by origin era (higher is better) ===")
    print(scores.round(3).to_string())
    print("\n=== MAE, bias, skill by target ===")
    print(details.round(4).to_string(index=False))
    if quick:
        return
    boot = mc.bootstrap_vs(wide, models, ref="team_selected")
    print("\n=== Bootstrap vs. team ===")
    print(boot.loc[scores.index].round(3).to_string())
    os.makedirs(RESULTS, exist_ok=True)
    scores.to_csv(os.path.join(RESULTS, "ensemble_scores_by_era.csv"))
    details.to_csv(os.path.join(RESULTS, "ensemble_metrics_by_target.csv"), index=False)
    era_detail.to_csv(os.path.join(RESULTS, "ensemble_metrics_by_target_era.csv"), index=False)
    boot.to_csv(os.path.join(RESULTS, "ensemble_bootstrap_vs_team.csv"))
    availability.to_csv(os.path.join(RESULTS, "ensemble_component_coverage.csv"))
    (wide.groupby(["target", "origin"]).size().rename("n_rows")
       .to_csv(os.path.join(RESULTS, "ensemble_evaluation_coverage.csv")))
    print(f"\nSaved results and coverage reports in {RESULTS}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Two origins; print only, do not save")
    args = parser.parse_args()
    start = time.time()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ns = mc.load_team_pipeline()
    print(f"Team pipeline loaded in {time.time()-start:.0f}s", flush=True)
    origins = [2004, 2011] if args.quick else [o for o in ns["ORIGINS"] if FIRST_ORIGIN <= o <= 2011]
    wide, availability = collect_predictions(ns, origins)
    evaluate(wide, availability, quick=args.quick)
    print(f"Finished in {time.time()-start:.0f}s", flush=True)


if __name__ == "__main__":
    main()