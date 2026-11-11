"""Backtest ARIMA and related time-series models on the same folds as model_comparison.py.

Data up to 2018 only. Every fit, stationarity test and order choice for an origin reads
only values dated <= that origin, over the trailing 40 calendar years. Gaps stay as NaN,
which ARIMA's state-space form handles directly.

Stationarity decides differencing in each fold, combining ADF and KPSS (trend-stationary null):
    ADF rejects, KPSS does not -> trend-stationary, d = 0
    ADF does not, KPSS rejects -> difference-stationary, d = 1
    both or neither reject     -> not settled, d = 0 with a linear trend available
AICc then picks the order among models with that d, because AIC is not comparable across d.

Run from the repository root:
    pip install datasets pandas numpy matplotlib scikit-learn statsmodels joblib
    python code/experiments/arima_comparison.py            # full run
    python code/experiments/arima_comparison.py --quick    # two origins, for a smoke test
"""
import argparse
import os
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss

import model_comparison as mc

HORIZON, WINDOW, MIN_VALID, ALPHA = 7, 40, 20, 0.05
ORDERS_D0 = {"mean_only": ((0, 0, 0), "c"), "arima_100_mean": ((1, 0, 0), "c"), "linear_trend": ((0, 0, 0), "ct"),
             "arima_100_trend": ((1, 0, 0), "ct"), "arima_001_trend": ((0, 0, 1), "ct"),
             "arima_101_trend": ((1, 0, 1), "ct")}
ORDERS_D1 = {"rw_drift": ((0, 1, 0), "t"), "arima_011": ((0, 1, 1), "n"), "arima_011_drift": ((0, 1, 1), "t"),
             "arima_111": ((1, 1, 1), "n")}
EXTRA = {"arima_022": ((0, 2, 2), "n")}  # equivalent to a local linear trend; not offered to AICc
TS_MODELS = list(ORDERS_D0) + list(ORDERS_D1) + list(EXTRA) + ["auto_arima", "auto_arima_log", "theta"]


def classify(v):
    """Combine ADF and KPSS, both allowing a linear trend. Returns (class, adf rejects, kpss rejects)."""
    if np.ptp(v) == 0:
        return "not_settled", False, False
    adf_p = adfuller(v, regression="ct", autolag="AIC", maxlag=3)[1]
    kpss_p = kpss(v, regression="ct", nlags="auto")[1]
    a, k = adf_p < ALPHA, kpss_p < ALPHA
    if a and not k:
        return "trend_stationary", a, k
    if k and not a:
        return "difference_stationary", a, k
    return "not_settled", a, k


def fit(v, spec):
    order, trend = spec
    try:
        res = ARIMA(v, order=order, trend=trend).fit()
        f = np.asarray(res.forecast(HORIZON), dtype=float)
        if np.isfinite(f).all() and np.isfinite(res.aicc):
            return f, res.aicc, bool(res.mle_retvals.get("converged", True))
    except Exception:
        pass
    return None, np.inf, False


def auto_select(v, cls, cache=None):
    """Lowest-AICc model among those with the differencing the stationarity tests chose."""
    group = ORDERS_D1 if cls == "difference_stationary" else ORDERS_D0
    fits = {m: (cache[m] if cache and m in cache else fit(v, s)) for m, s in group.items()}
    best = min(fits, key=lambda m: fits[m][1])
    return fits[best][0], best


def theta(v):
    """Classic Theta: smoothed level plus half the fitted linear drift."""
    v = v[~np.isnan(v)]
    grid = np.arange(0.05, 0.96, 0.05)
    level = np.full(len(grid), v[0])
    sae = np.zeros(len(grid))
    for x in v[1:]:
        sae += np.abs(x - level)
        level = level + grid * (x - level)
    k = int(np.argmin(sae))
    a, n = grid[k], len(v)
    slope = np.polyfit(np.arange(n), v, 1)[0]
    h = np.arange(1, HORIZON + 1)
    return level[k] + 0.5 * slope * (h - 1 + (1 - (1 - a) ** n) / a)


def run_series(window, log_too):
    """Forecast one city at one origin. The window ends at the origin; returns None if too sparse."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = np.argmax(~np.isnan(window)) if (~np.isnan(window)).any() else len(window)
        y = window[first:]
        v = y[~np.isnan(y)]
        if len(v) < MIN_VALID:
            return None
        cls, adf_rej, kpss_rej = classify(v)
        fits = {m: fit(y, s) for m, s in {**ORDERS_D0, **ORDERS_D1, **EXTRA}.items()}
        out = {m: f[0] for m, f in fits.items()}
        out["auto_arima"], picked = auto_select(y, cls, fits)
        out["theta"] = theta(y)
        if log_too:
            ly, lv = np.log1p(y), np.log1p(v)
            lf, _ = auto_select(ly, classify(lv)[0])
            out["auto_arima_log"] = np.expm1(lf) if lf is not None else None
        else:
            out["auto_arima_log"] = out["auto_arima"]
        return {"f": out, "class": cls, "adf": adf_rej, "kpss": kpss_rej, "picked": picked,
                "nonconv": sum(not f[2] for f in fits.values()), "failed": sum(f[0] is None for f in fits.values())}


def factor_model(X, o, yidx, k=3):
    """Regional model: a few factors shared across cities, each projected with AR(1) around a trend."""
    oi = yidx(o)
    W = X[:, oi - WINDOW + 1:oi + 1]
    ok = (~np.isnan(W)).sum(1) >= MIN_VALID
    mean = np.nanmean(W[ok], axis=1, keepdims=True)
    A = np.nan_to_num(W[ok] - mean)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    t = np.arange(WINDOW)
    tf = np.arange(WINDOW, WINDOW + HORIZON)
    F = np.zeros((k, HORIZON))
    for j in range(k):
        s, c = np.polyfit(t, Vt[j], 1)
        r = Vt[j] - (s * t + c)
        phi = float(np.clip(r[:-1] @ r[1:] / max(r[:-1] @ r[:-1], 1e-12), 0, 0.9))
        F[j] = s * tf + c + phi ** np.arange(1, HORIZON + 1) * r[-1]
    out = np.full((X.shape[0], HORIZON), np.nan)
    out[ok] = mean + (U[:, :k] * S[:k]) @ F
    return out


def predict_origin(X, o, ns, cache, nonneg, pool):
    oi = ns["yidx"](o)
    jobs = [delayed(run_series)(X[i, max(0, oi - WINDOW + 1):oi + 1].copy(), nonneg) for i in range(X.shape[0])]
    res = pool(jobs)
    P = {m: np.full((X.shape[0], HORIZON), np.nan) for m in TS_MODELS}
    meta = []
    for i, r in enumerate(res):
        if r is None:
            continue
        for m, f in r["f"].items():
            if f is not None:
                P[m][i] = f
        meta.append({"origin": o, "city": i, "class": r["class"], "adf_rejects": r["adf"], "kpss_rejects": r["kpss"],
                     "picked": r["picked"], "nonconverged_fits": r["nonconv"], "failed_fits": r["failed"]})
    P["factor_3"] = factor_model(X, o, ns["yidx"])
    base = mc.statistical_models(X, o, ns["years"], ns["yidx"])
    for m in ["clim30", "last_year", "damped_holt"]:
        P[m] = base[m]
    P.update(mc.ml_models(X, o, cache, names=["ridge"]))
    return {m: np.maximum(v, 0.0) if nonneg else v for m, v in P.items()}, meta


def leakage_test(ns, pool, o=2004):
    X = np.where(ns["STRICT"]["cdd"], ns["A"]["cdd"], np.nan)
    ref, _ = predict_origin(X, o, ns, mc.FeatureCache(X, ns), True, pool)
    Y = X.copy()
    cut = ns["yidx"](o) + 1
    Y[:, cut:] = np.random.default_rng(1).normal(0, 1e4, Y[:, cut:].shape)
    alt, _ = predict_origin(Y, o, ns, mc.FeatureCache(Y, ns), True, pool)
    for m in ref:
        assert np.allclose(ref[m], alt[m], equal_nan=True), "LEAK in " + m
    print(f"Leakage test passed for {len(ref)} models at origin {o}.")


def stationarity_at(X, o, ns):
    """ADF and KPSS using each city's trailing window at one origin, without any forecasting."""
    oi = ns["yidx"](o)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(X.shape[0]):
            w = X[i, max(0, oi - WINDOW + 1):oi + 1]
            v = w[~np.isnan(w)]
            if len(v) >= MIN_VALID:
                c, a, k = classify(v)
                rows.append({"class": c, "adf_rejects": a, "kpss_rejects": k})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="two origins only, results not saved")
    args = ap.parse_args()
    t0 = time.time()
    ns = mc.load_team_pipeline()
    print(f"Team pipeline loaded ({time.time()-t0:.0f}s).")
    targets, bt, selection = ns["TARGETS"], ns["bt"], ns["selection"]
    origins = [2004, 2011] if args.quick else [o for o in ns["ORIGINS"] if o >= 1990]

    long, meta = [], []
    with Parallel(n_jobs=-1, batch_size=16) as pool:
        leakage_test(ns, pool)
        for t in targets:
            X = np.where(ns["STRICT"][t], ns["A"][t], np.nan)
            cache = mc.FeatureCache(X, ns)
            for o in origins:
                P, m = predict_origin(X, o, ns, cache, t in ns["NONNEG"], pool)
                meta += [dict(r, target=t) for r in m]
                oi = ns["yidx"](o)
                truth = X[:, oi + 1:oi + 1 + HORIZON]
                ci, hi = np.nonzero(~np.isnan(truth))
                frame = {"origin": o, "target": t, "city": ci, "year": o + hi + 1, "actual": truth[ci, hi]}
                frame.update({k: v[ci, hi] for k, v in P.items()})
                long.append(pd.DataFrame(frame))
            print(f"{t}: done ({time.time()-t0:.0f}s)")
    long, meta = pd.concat(long, ignore_index=True), pd.DataFrame(meta)

    key = ["origin", "target", "city", "year"]
    team = bt[bt.method == bt.target.map(selection)][key + ["predicted"]].rename(columns={"predicted": "team_selected"})
    wide = long.merge(team, on=key)
    ts_cols = [m for m in wide.columns if m not in key + ["actual"]]
    print("Rows with a missing forecast, by model:")
    missing = wide[ts_cols].isna().sum()
    print(missing[missing > 0].to_string() if (missing > 0).any() else "none")
    before = len(wide)
    wide = wide.dropna()
    print(f"Matched rows: {len(wide):,} of {before:,}. Failed fits: {int(meta.failed_fits.sum())}; "
          f"non-converged fits: {int(meta.nonconverged_fits.sum())} of {len(meta) * 11:,}.")

    for name, other in [("ens_team_auto", "auto_arima"), ("ens_team_auto_log", "auto_arima_log"),
                        ("ens_team_theta", "theta"), ("ens_team_factor", "factor_3"), ("ens_team_ridge", "ridge")]:
        wide[name] = (wide["team_selected"] + wide[other]) / 2
    models = ["team_selected", "last_year", "damped_holt", "ridge", "factor_3"] + \
             [m for m in TS_MODELS if m != "mean_only"] + \
             ["ens_team_auto", "ens_team_auto_log", "ens_team_theta", "ens_team_factor", "ens_team_ridge"]

    pd.set_option("display.width", 170, "display.max_columns", 30, "display.max_rows", 200)
    scores = mc.era_scores(wide, models)
    print("\n=== Score-style skill by origin era ===")
    print(scores.round(2))
    detail = mc.score_table(wide, models)
    print("\n=== Skill by metric ===")
    print(detail["skill"].unstack("target").loc[scores.index].round(3))
    print("\n=== Bias by metric, origins 2005-2011 ===")
    print(mc.score_table(wide[wide.origin >= 2005], models)["bias"].unstack("target").loc[scores.index].round(3))
    print("\n=== Stationarity class share by metric, all folds ===")
    print(meta.groupby("target")["class"].value_counts(normalize=True).unstack().round(3))
    print("\n=== Order chosen by auto_arima, share by metric ===")
    print(meta.groupby("target")["picked"].value_counts(normalize=True).unstack().fillna(0).round(3))

    stat_rows = []
    for t in targets:
        d = stationarity_at(np.where(ns["STRICT"][t], ns["A"][t], np.nan), 2018, ns)
        stat_rows.append({"target": t, "cities": len(d), "adf_rejects": d.adf_rejects.mean(), "kpss_rejects": d.kpss_rejects.mean(),
                          **d["class"].value_counts(normalize=True).to_dict()})
    stat = pd.DataFrame(stat_rows).set_index("target").fillna(0)
    print("\n=== Stationarity at the 2018 origin (trailing 40 years) ===")
    print(stat.round(3))

    if args.quick:
        print(f"\nQuick run finished in {time.time()-t0:.0f}s; nothing saved.")
        return

    boot = mc.bootstrap_vs(wide, models, "team_selected")
    print("\n=== Bootstrap over calendar years, origins 1990-2011 ===")
    print(boot.loc[scores.index].round(3))

    os.makedirs(mc.RESULTS, exist_ok=True)
    scores.round(3).to_csv(os.path.join(mc.RESULTS, "arima_scores_by_era.csv"))
    detail.round(4).to_csv(os.path.join(mc.RESULTS, "arima_metrics_1990_2011.csv"))
    boot.round(4).to_csv(os.path.join(mc.RESULTS, "arima_bootstrap_vs_team.csv"))
    stat.round(4).to_csv(os.path.join(mc.RESULTS, "stationarity_at_2018.csv"))
    print(f"\nFinished in {time.time()-t0:.0f}s. Results in {mc.RESULTS}")


if __name__ == "__main__":
    main()
