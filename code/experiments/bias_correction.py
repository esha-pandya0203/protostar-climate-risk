"""Test whether a walk-forward bias correction improves the team's forecasts (data up to 2018 only).

Motivation: the backtests show the team's forecasts run too cold in recent years. At each origin o
we measure how far the team's earlier forecasts missed, using ONLY blocks fully observed by o
(origin' + 7 <= o), and add that average miss back to the forecast for o+1..o+7.

Variants (all computed per target and per horizon step h = 1..7):
    bias_global  team + mean past miss across all cities
    bias_half    team + 0.5 * that miss (shrinks toward no correction)
    bias_region  team + each city's past miss, median-pooled over the city and its neighbours

Run from the repository root, next to model_comparison.py:
    python code/experiments/bias_correction.py

"""
import os
import time
import warnings

import numpy as np
import pandas as pd

import model_comparison as mc

HORIZON, MIN_PAST_BLOCKS, EVAL_FIRST = 7, 5, 1990

def clim30(X, o, ns):
    """30-year average of the values dated <= o, needing at least 10 valid years (as in mc)."""
    oi = ns["yidx"](o)
    W = X[:, max(0, oi - 29):oi + 1]
    ok = (~np.isnan(W)).sum(1) >= 10
    out = np.full(X.shape[0], np.nan)
    out[ok] = np.nanmean(W[ok], axis=1)
    return out


def target_frame(t, ns, bt, selection):
    """One row per (origin, city, year) with the team forecast, the actual and the clim30 baseline."""
    X = np.where(ns["STRICT"][t], ns["A"][t], np.nan)
    parts = []
    for o in ns["ORIGINS"]:
        oi = ns["yidx"](o)
        truth = X[:, oi + 1:oi + 1 + HORIZON]
        if truth.shape[1] < HORIZON:
            truth = np.pad(truth, ((0, 0), (0, HORIZON - truth.shape[1])), constant_values=np.nan)
        ci, hi = np.nonzero(~np.isnan(truth))
        parts.append(pd.DataFrame({"origin": o, "target": t, "city": ci, "year": o + hi + 1, "actual": truth[ci, hi],
                                   "clim30": clim30(X, o, ns)[ci]}))
    long = pd.concat(parts, ignore_index=True)
    key = ["origin", "target", "city", "year"]
    team = bt[(bt.target == t) & (bt.method == selection[t])][key + ["predicted"]].rename(
        columns={"predicted": "team_selected"})
    return long.merge(team, on=key).dropna()


def correct_target(g, ns, nonneg):
    """Add walk-forward corrections to one target's frame. Uses only blocks fully observed by each origin."""
    origins = np.array(sorted(g.origin.unique()))
    n = len(ns["nbr"])
    R = np.full((len(origins), n, HORIZON), np.nan)           # actual - team, by origin, city, horizon step
    oi = np.searchsorted(origins, g.origin.to_numpy())
    h = (g.year - g.origin - 1).to_numpy()
    R[oi, g.city.to_numpy(), h] = (g.actual - g.team_selected).to_numpy()
    nbr = ns["nbr"]

    out = []
    for o in origins[origins >= EVAL_FIRST]:
        past = R[origins + HORIZON <= o]
        sub = g[g.origin == o].copy()
        hh, cc = (sub.year - o - 1).to_numpy(), sub.city.to_numpy()
        if len(past) < MIN_PAST_BLOCKS:
            for m in ["bias_global", "bias_half", "bias_region"]:
                sub[m] = sub.team_selected
            out.append(sub)
            continue
        city_miss = np.nanmean(past, axis=0)                   # (city, h)
        glob = np.nanmean(city_miss, axis=0)                   # (h,)
        reg = np.array([[np.nanmedian(city_miss[np.r_[i, nbr[i]], k]) for k in range(HORIZON)] for i in range(n)])
        sub["bias_global"] = sub.team_selected + glob[hh]
        sub["bias_half"] = sub.team_selected + 0.5 * glob[hh]
        sub["bias_region"] = sub.team_selected + np.nan_to_num(reg[cc, hh])
        out.append(sub)
    res = pd.concat(out, ignore_index=True)
    # No leakage by construction: every block used had origin' + 7 <= o, so year <= o.
    if nonneg:
        for m in ["bias_global", "bias_half", "bias_region"]:
            res[m] = np.maximum(res[m], 0.0)
    return res


def main():
    t0 = time.time()
    warnings.filterwarnings("ignore")
    ns = mc.load_team_pipeline()
    bt, selection = ns["bt"], ns["selection"]
    print(f"Team pipeline loaded ({time.time()-t0:.0f}s).")

    frames = []
    for t in ns["TARGETS"]:
        g = target_frame(t, ns, bt, selection)
        frames.append(correct_target(g, ns, t in ns["NONNEG"]))
        print(f"{t}: done ({time.time()-t0:.0f}s)")
    wide = pd.concat(frames, ignore_index=True)
    wide = wide[wide.origin >= EVAL_FIRST]
    models = ["team_selected", "bias_global", "bias_half", "bias_region"]

    pd.set_option("display.width", 170, "display.max_columns", 30)
    scores = mc.era_scores(wide, models)
    print("\n=== Score-style skill by origin era ===")
    print(scores.round(2))
    detail = mc.score_table(wide, models)
    print("\n=== Skill by metric ===")
    print(detail["skill"].unstack("target").loc[scores.index].round(3))
    print("\n=== Bias by metric, origins 2005-2011 ===")
    print(mc.score_table(wide[wide.origin >= 2005], models)["bias"].unstack("target").loc[scores.index].round(3))
    boot = mc.bootstrap_vs(wide, models, "team_selected")
    print("\n=== Bootstrap over calendar years vs team ===")
    print(boot.loc[scores.index].round(3))

    os.makedirs(mc.RESULTS, exist_ok=True)
    scores.round(3).to_csv(os.path.join(mc.RESULTS, "bias_correction_scores.csv"))
    boot.round(4).to_csv(os.path.join(mc.RESULTS, "bias_correction_bootstrap.csv"))
    print(f"\nFinished in {time.time()-t0:.0f}s.")


if __name__ == "__main__":
    main()