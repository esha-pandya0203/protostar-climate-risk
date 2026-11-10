"""Compare forecasting models with rolling-origin cross-validation (data up to 2018 only).

Each fold stands at an origin year o, trains on information dated <= o, and predicts
o+1..o+7, the same 7-year block shape as the challenge. Random k-fold is not used because
it would train on years later than the ones being predicted.

The team's forecasters come from Protostar_Final_Pipeline.ipynb: its code cells are executed
unchanged up to the selection step, so they are scored exactly as the notebook defines them.

Scoring mirrors the challenge: skill = 1 - MAE / MAE(30-year average), averaged over the four
metrics, on strict official values (at most 10 missing days). Skill here is not floored at 0.

Run from the repository root:
    pip install datasets pandas numpy matplotlib scikit-learn
    python code/experiments/model_comparison.py            # full run
    python code/experiments/model_comparison.py --quick    # two origins, for a smoke test
"""
import argparse
import itertools
import json
import os
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(ROOT, "Protostar_Final_Pipeline.ipynb")
RESULTS = os.path.join(ROOT, "experiments", "results")
TRAIN_START = 1950   # earliest origin used to train the pooled ML models
HORIZON = 7
ML_MODELS = ["ridge", "random_forest", "gradient_boosting"]
N_BOOT = 2000


def load_team_pipeline():
    """Execute the notebook's code cells up to and including the selection step."""
    import matplotlib
    matplotlib.use("Agg")
    nb = json.load(open(NOTEBOOK, encoding="utf-8"))
    ns = {"display": lambda *a, **k: None}
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)  # the notebook creates ./data
        try:
            for cell in nb["cells"]:
                src = "".join(cell["source"])
                if cell["cell_type"] != "code" or src.startswith("!pip"):
                    continue
                exec(compile(src, "notebook", "exec"), ns)
                if "selection, tables = {}" in src:
                    break
        finally:
            os.chdir(cwd)
    assert ns["CUTOFF"] == 2018 and ns["annual"]["Year"].max() <= 2018, "data rule violated"
    return ns


# ---------------------------------------------------------------- per-city statistical models
def holt_forecast(v, alphas, betas, phis):
    """Exponential smoothing with parameters picked by in-sample one-step absolute error."""
    grid = np.array(list(itertools.product(alphas, betas, phis)))
    a, b_, p = grid[:, 0], grid[:, 1], grid[:, 2]
    level = np.full(len(grid), v[0], dtype=float)
    slope = np.where(b_ > 0, np.full(len(grid), np.mean(np.diff(v[:6]))), 0.0)
    sae = np.zeros(len(grid))
    for x in v[1:]:
        fit = level + p * slope
        err = x - fit
        sae += np.abs(err)
        new_level = fit + a * err
        slope = b_ * (new_level - level) + (1 - b_) * p * slope
        level = new_level
    k = int(np.argmin(sae))
    damp = np.cumsum(p[k] ** np.arange(1, HORIZON + 1))
    return level[k] + damp * slope[k]


def statistical_models(X, o, years, yidx):
    """Per-city forecasts. Every model reads only X[:, :origin+1]."""
    n = X.shape[0]
    oi = yidx(o)
    T = np.arange(o + 1, o + HORIZON + 1)
    names = ["clim30", "last_year", "ses", "damped_holt", "theil_sen30", "ar1_trend30"]
    out = {m: np.full((n, HORIZON), np.nan) for m in names}
    yy = years[:oi + 1]
    for i in range(n):
        m = ~np.isnan(X[i, :oi + 1])
        if not m.any():
            continue
        y, v = yy[m], X[i, :oi + 1][m]
        out["last_year"][i] = v[-1]
        w = y > o - 30
        if w.sum() >= 10:
            yw, vw = y[w], v[w]
            out["clim30"][i] = vw.mean()
            ii, jj = np.triu_indices(len(yw), 1)
            ts = np.median((vw[jj] - vw[ii]) / (yw[jj] - yw[ii]))
            out["theil_sen30"][i] = np.median(vw - ts * yw) + ts * T
            s, c = np.polyfit(yw, vw, 1)
            r = vw - (s * yw + c)
            cons = np.diff(yw) == 1
            den = (r[:-1][cons] ** 2).sum()
            phi = float(np.clip((r[:-1][cons] * r[1:][cons]).sum() / den, 0, 0.9)) if den > 0 else 0.0
            out["ar1_trend30"][i] = s * T + c + phi ** (T - yw[-1]) * r[-1]
        recent = v[y > o - 40]
        if len(recent) >= 10:
            out["ses"][i] = holt_forecast(recent, [0.05, 0.1, 0.2, 0.3, 0.5], [0.0], [0.0])
            out["damped_holt"][i] = holt_forecast(recent, [0.05, 0.1, 0.2, 0.3, 0.5],
                                                  [0.01, 0.05, 0.1], [0.8, 0.9, 0.98])
    return out


# ---------------------------------------------------------------- pooled ML models
FEATURES = ["last_a", "m3_a", "m5_a", "m10_a", "med30_a", "std30", "slope30", "pooled_slope",
            "n30", "staleness", "nat_a", "nbr_a", "clim30", "lat", "lon"]


def origin_features(X, t, years, yidx, nbr, lat, lon):
    """Features known at the end of year t. Levels are anomalies from the 30-year average."""
    n = X.shape[0]
    ti = yidx(t)
    yy = years[:ti + 1]
    F = np.full((n, len(FEATURES)), np.nan)
    for i in range(n):
        m = ~np.isnan(X[i, :ti + 1])
        y, v = yy[m], X[i, :ti + 1][m]
        w = y > t - 30
        if w.sum() < 10:
            continue
        clim = v[w].mean()
        F[i, :7] = [v[-1] - clim, v[-3:].mean() - clim, v[-5:].mean() - clim, v[-10:].mean() - clim,
                    np.median(v[w]) - clim, v[w].std(), np.polyfit(y[w], v[w], 1)[0]]
        F[i, 8:10] = [w.sum(), t - y[-1]]
        F[i, 12] = clim
    F[:, 7] = [np.nanmedian(F[np.r_[i, nbr[i]], 6]) if np.isfinite(F[np.r_[i, nbr[i]], 6]).any() else np.nan
               for i in range(n)]
    F[:, 10] = np.nanmedian(F[:, 2])
    F[:, 11] = [np.nanmedian(F[nbr[i], 2]) if np.isfinite(F[nbr[i], 2]).any() else np.nan for i in range(n)]
    F[:, 13], F[:, 14] = lat, lon
    return F


def with_horizon(F, h):
    """Add the horizon and its interaction with the two trend features."""
    return np.column_stack([F, np.full(len(F), h), h * F[:, 6], h * F[:, 7]])


def make_ml(name):
    if name == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100.0))
    if name == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestRegressor(n_estimators=100, min_samples_leaf=25, max_features=0.5,
                                                   max_samples=0.3, n_jobs=-1, random_state=0))
    return HistGradientBoostingRegressor(loss="absolute_error", max_iter=150, learning_rate=0.05,
                                         max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0,
                                         random_state=0)


def ml_models(X, o, feats, names=ML_MODELS):
    """Train on (features at t, value at t+h) pairs with t+h <= o, then predict o+1..o+7."""
    n = X.shape[0]
    Xtr, ytr = [], []
    for t in range(TRAIN_START, o):
        F = feats(t)
        for h in range(1, HORIZON + 1):
            if t + h > o:
                break
            target = X[:, t + h - feats.year0] - F[:, 12]  # anomaly from the 30-year average at t
            keep = np.isfinite(target)
            Xtr.append(with_horizon(F, h)[keep])
            ytr.append(target[keep])
    Xtr, ytr = np.vstack(Xtr), np.concatenate(ytr)
    F0 = feats(o)
    Xte = np.vstack([with_horizon(F0, h) for h in range(1, HORIZON + 1)])
    ok = np.isfinite(F0[:, 12])
    out = {}
    for name in names:
        model = make_ml(name).fit(Xtr, ytr)
        pred = model.predict(Xte).reshape(HORIZON, n).T + F0[:, 12:13]
        pred[~ok] = np.nan
        out[name] = pred
    return out


class FeatureCache:
    def __init__(self, X, ns):
        self.X, self.ns, self.c, self.year0 = X, ns, {}, ns["YEAR0"]

    def __call__(self, t):
        if t not in self.c:
            ns = self.ns
            self.c[t] = origin_features(self.X, t, ns["years"], ns["yidx"], ns["nbr"], ns["la"], ns["lo"])
        return self.c[t]


def predict_all(X, o, ns, cache, nonneg):
    P = statistical_models(X, o, ns["years"], ns["yidx"])
    P.update(ml_models(X, o, cache))
    return {m: np.maximum(v, 0.0) if nonneg else v for m, v in P.items()}


def leakage_test(ns, o=2004):
    """Scrambling every value after the origin must change no prediction."""
    t = "cdd"
    X = np.where(ns["STRICT"][t], ns["A"][t], np.nan)
    ref = predict_all(X, o, ns, FeatureCache(X, ns), True)
    Y = X.copy()
    cut = ns["yidx"](o) + 1
    Y[:, cut:] = np.random.default_rng(1).normal(0, 1e4, Y[:, cut:].shape)
    alt = predict_all(Y, o, ns, FeatureCache(Y, ns), True)
    for m in ref:
        assert np.allclose(ref[m], alt[m], equal_nan=True), "LEAK in " + m
    print(f"Leakage test passed for {len(ref)} models at origin {o}.")


# ---------------------------------------------------------------- scoring
def score_table(df, models):
    rows = {}
    for t, g in df.groupby("target"):
        base = (g["clim30"] - g["actual"]).abs().mean()
        for m in models:
            e = g[m] - g["actual"]
            rows[(t, m)] = {"mae": e.abs().mean(), "bias": e.mean(), "skill": 1 - e.abs().mean() / base}
    tab = pd.DataFrame(rows).T
    tab.index.names = ["target", "model"]
    return tab


def era_scores(df, models):
    eras = {"1990-2011": df.origin >= 1990, "1990-1999": df.origin.between(1990, 1999),
            "2000-2011": df.origin >= 2000, "2005-2011": df.origin >= 2005}
    res = pd.DataFrame({k: 100 * score_table(df[v], models)["skill"].groupby("model").mean()
                        for k, v in eras.items() if v.any()})
    return res.sort_values("1990-2011", ascending=False)


def bootstrap_vs(df, models, ref, seed=0):
    """Resample calendar years, because errors are strongly correlated across cities within a year."""
    rng = np.random.default_rng(seed)
    yrs = np.sort(df.year.unique())
    W = rng.multinomial(len(yrs), np.ones(len(yrs)) / len(yrs), size=N_BOOT).astype(float)
    per_target = {}
    for t, g in df.groupby("target"):
        sums = {m: (g[m] - g["actual"]).abs().groupby(g["year"]).sum().reindex(yrs).fillna(0).to_numpy()
                for m in models + ["clim30"]}
        base = W @ sums["clim30"]
        per_target[t] = {m: 1 - (W @ sums[m]) / base for m in models}
    out = []
    for m in models:
        s = 100 * np.mean([per_target[t][m] for t in per_target], axis=0)
        r = 100 * np.mean([per_target[t][ref] for t in per_target], axis=0)
        d = s - r
        out.append({"model": m, "score_lo": np.quantile(s, 0.025), "score_hi": np.quantile(s, 0.975),
                    "diff_vs_team": d.mean(), "diff_lo": np.quantile(d, 0.025), "diff_hi": np.quantile(d, 0.975),
                    "P(beats team)": (d > 0).mean()})
    return pd.DataFrame(out).set_index("model")


def nested_selection(wide, models, first=2000):
    """At each origin pick each metric's model using only blocks fully observed by that origin."""
    picks, parts = [], []
    for o in sorted(o for o in wide.origin.unique() if o >= first):
        for t, g in wide[wide.origin <= o].groupby("target"):
            past = g[g.origin + HORIZON <= o]
            mae = {m: (past[m] - past["actual"]).abs().mean() for m in models}
            best = min(mae, key=mae.get)
            picks.append({"origin": o, "target": t, "picked": best})
            test = g[g.origin == o]
            parts.append(test.assign(nested=test[best]))
    return pd.concat(parts, ignore_index=True), pd.DataFrame(picks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="two origins only, results not saved")
    args = ap.parse_args()
    t0 = time.time()
    ns = load_team_pipeline()
    print(f"Team pipeline loaded ({time.time()-t0:.0f}s). Selected: {ns['selection']}")
    leakage_test(ns)

    targets, bt, selection = ns["TARGETS"], ns["bt"], ns["selection"]
    origins = [2004, 2011] if args.quick else list(ns["ORIGINS"])
    long = []
    for t in targets:
        X = np.where(ns["STRICT"][t], ns["A"][t], np.nan)
        cache = FeatureCache(X, ns)
        for o in origins:
            P = predict_all(X, o, ns, cache, t in ns["NONNEG"])
            oi = ns["yidx"](o)
            truth = X[:, oi + 1:oi + 1 + HORIZON]
            ci, hi = np.nonzero(~np.isnan(truth))
            frame = {"origin": o, "target": t, "city": ci, "year": o + hi + 1, "actual": truth[ci, hi]}
            frame.update({m: P[m][ci, hi] for m in P})
            long.append(pd.DataFrame(frame))
        print(f"{t}: done ({time.time()-t0:.0f}s)")
    long = pd.concat(long, ignore_index=True)

    key = ["origin", "target", "city", "year"]
    team = bt[bt.method == bt.target.map(selection)][key + ["predicted"]].rename(columns={"predicted": "team_selected"})
    bases = (bt[bt.method.isin(["base_recent10", "base_trend30"])].pivot_table(index=key, columns="method", values="predicted")
             .rename(columns={"base_recent10": "recent10", "base_trend30": "trend30"}).reset_index())
    wide = long.merge(team, on=key).merge(bases, on=key)
    before = len(wide)
    wide = wide.dropna()
    print(f"Matched rows: {len(wide):,} of {before:,} (every model is scored on identical rows).")

    wide["ens_team_gb"] = (wide["team_selected"] + wide["gradient_boosting"]) / 2
    wide["ens_team_ridge"] = (wide["team_selected"] + wide["ridge"]) / 2
    wide["ens_team_ml"] = wide[["team_selected"] + ML_MODELS].mean(axis=1)
    models = ["team_selected", "recent10", "trend30", "last_year", "ses", "damped_holt", "theil_sen30",
              "ar1_trend30"] + ML_MODELS + ["ens_team_gb", "ens_team_ridge", "ens_team_ml"]

    pd.set_option("display.width", 170, "display.max_columns", 30, "display.max_rows", 200)
    eval_df = wide[wide.origin >= 1990]
    scores = era_scores(wide, models)
    print("\n=== Score-style skill (100 x mean of 1 - MAE/MAE_clim30) by origin era ===")
    print(scores.round(2))
    detail = score_table(eval_df, models)
    print("\n=== Skill by metric, origins 1990-2011 ===")
    print(detail["skill"].unstack("target").loc[scores.index].round(3))
    print("\n=== MAE by metric, origins 1990-2011 ===")
    print(detail["mae"].unstack("target").loc[scores.index].round(3))
    print("\n=== Bias by metric, origins 2005-2011 ===")
    print(score_table(wide[wide.origin >= 2005], models)["bias"].unstack("target").loc[scores.index].round(3))

    if args.quick:
        print(f"\nQuick run finished in {time.time()-t0:.0f}s; nothing saved.")
        return

    boot = bootstrap_vs(eval_df, models, "team_selected")
    print("\n=== Bootstrap over calendar years, origins 1990-2011 ===")
    print(boot.loc[scores.index].round(3))

    # The notebook picked team_selected using these same folds, so it carries some selection optimism.
    nested_pool = [m for m in models if m != "team_selected"]
    nested, picks = nested_selection(wide, nested_pool)
    print("\n=== Walk-forward selection, origins 2000-2011 (picks use only past blocks; team_selected excluded) ===")
    print(era_scores(nested, ["nested", "team_selected"])[["2000-2011", "2005-2011"]].round(2))
    print(picks.groupby(["target", "picked"]).size().to_string())

    os.makedirs(RESULTS, exist_ok=True)
    scores.round(3).to_csv(os.path.join(RESULTS, "scores_by_era.csv"))
    detail.round(4).to_csv(os.path.join(RESULTS, "metrics_1990_2011.csv"))
    boot.round(4).to_csv(os.path.join(RESULTS, "bootstrap_vs_team.csv"))
    picks.to_csv(os.path.join(RESULTS, "walk_forward_picks.csv"), index=False)
    print(f"\nFinished in {time.time()-t0:.0f}s. Results in {RESULTS}")


if __name__ == "__main__":
    main()
