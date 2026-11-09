"""Experimental improvements to the Protostar 2018-cutoff forecasting pipeline.

Run from the project root:
    python experiments/error_analysis.py --diagnostics-only
    python experiments/error_analysis.py --quick
    python experiments/error_analysis.py

This imports the existing notebook via model_comparison.load_team_pipeline().
It does NOT edit the final notebook or submission.csv. No post-2018 data is loaded.
Candidate results are exploratory: team_selected was picked on overlapping historical folds.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import model_comparison as mc

RESULTS = Path(__file__).resolve().parent / "results" / "error_analysis_results"
HORIZON = 7
TREND_TARGETS = ("Avg Temp", "cdd")
KEY = ["origin", "target", "city", "year"]


def selected_team_rows(ns: dict, origins: list[int]) -> pd.DataFrame:
    """Reuse the baseline notebook's exact held-out predictions and truth values."""
    bt = ns["bt"]
    selected = ns["selection"]
    x = bt[(bt["origin"].isin(origins)) & (bt["method"] == bt["target"].map(selected))]
    x = x[KEY + ["horizon", "actual", "predicted"]].copy()
    x = x.rename(columns={"predicted": "team_selected"})
    assert not x.duplicated(KEY).any(), "Duplicate team forecast keys"
    return x


def diagnostics(team: pd.DataFrame, ns: dict, folder: Path) -> None:
    """Diagnostic errors for the team's selected historical forecasts."""
    folder.mkdir(parents=True, exist_ok=True)
    z = team.copy()
    z["error"] = z.team_selected - z.actual
    z["abs_error"] = z.error.abs()

    def summarize(keys):
        return (z.groupby(keys, as_index=False)
                .agg(n=("error", "size"), mae=("abs_error", "mean"), bias=("error", "mean")))

    summarize(["target", "horizon"]).to_csv(folder / "error_by_horizon.csv", index=False)
    summarize(["target", "city"]).assign(
        city_id=lambda d: d.city.map(lambda i: str(ns["ids"][int(i)]))
    ).to_csv(folder / "error_by_city.csv", index=False)
    summarize(["target", "origin"]).to_csv(folder / "error_by_origin.csv", index=False)
    print("\n=== Forecast errors by horizon (team's selected models) ===")
    print(summarize(["target", "horizon"]).round(3).to_string(index=False))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for target, g in summarize(["target", "horizon"]).groupby("target"):
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            axes[0].plot(g.horizon, g.mae, marker="o")
            axes[0].set(title=f"{target}: MAE", xlabel="Forecast horizon (years)", ylabel="MAE")
            axes[1].plot(g.horizon, g.bias, marker="o")
            axes[1].axhline(0, color="gray", linewidth=1)
            axes[1].set(title=f"{target}: signed bias", xlabel="Forecast horizon (years)", ylabel="Predicted minus actual")
            fig.tight_layout()
            fig.savefig(folder / f"horizon_{target.replace(' ', '_')}.png", dpi=150)
            plt.close(fig)
    except ImportError:
        print("matplotlib not installed; CSV diagnostics were saved without plots")


def similarity_neighbors(X: np.ndarray, origin_idx: int, geographic_neighbors: np.ndarray,
                         k: int = 8, min_overlap: int = 15, window: int = 30) -> np.ndarray:
    """Choose top neighbors by *past* detrended annual anomaly correlation.

    Candidate pool is the precomputed closest geographic neighbors. Missing correlations
    are ranked last. Only history through origin_idx is examined.
    """
    n = X.shape[0]
    start = max(0, origin_idx - window + 1)
    past = X[:, start:origin_idx + 1].astype(float)
    result = np.empty((n, k), dtype=int)
    t = np.arange(past.shape[1], dtype=float)
    # Remove a city-specific historical linear trend before measuring co-fluctuation.
    anomaly = np.full_like(past, np.nan)
    for i, v in enumerate(past):
        valid = np.isfinite(v)
        if valid.sum() >= min_overlap:
            slope, intercept = np.polyfit(t[valid], v[valid], 1)
            anomaly[i, valid] = v[valid] - (slope * t[valid] + intercept)
    for i in range(n):
        candidates = geographic_neighbors[i]
        scores = []
        for j in candidates:
            ok = np.isfinite(anomaly[i]) & np.isfinite(anomaly[j])
            corr = np.corrcoef(anomaly[i, ok], anomaly[j, ok])[0, 1] if ok.sum() >= min_overlap else np.nan
            scores.append(corr if np.isfinite(corr) else -2.0)
        ranked = np.argsort(-np.asarray(scores), kind="stable")
        # Always return k; fallback to closest neighbors when correlations unavailable.
        chosen = candidates[ranked[:k]]
        result[i] = chosen
    return result


def experimental_forecasts(ns: dict, target: str, origin: int) -> dict[str, np.ndarray]:
    """Alternative trend forecasts, using the notebook's cleaned history at origin."""
    base = ns["predict"](target, origin)
    if target not in TREND_TARGETS:
        return {}
    x, _, _ = ns["homogenised"](target, origin)
    stats = ns["series_stats"](x, origin)
    oi = ns["yidx"](origin)
    future = np.arange(origin + 1, origin + HORIZON + 1)[None, :]
    gap = future - stats["ybar10"][:, None]
    old_slope = stats["pooled"]
    local_slope = stats["slope30"]
    # Shrink noisy city trends toward the existing geographic pooled trend.
    # A fixed shrinkage is deliberately used (no tuning on the evaluation years).
    local = np.where(np.isfinite(local_slope), local_slope, old_slope)
    adaptive = 0.25 * local + 0.75 * old_slope
    geo_pool = ns["order"]  # Already nearest-neighbor list from notebook
    similar = similarity_neighbors(x, oi, geo_pool, k=ns["K_NEIGHBORS"])
    sim_slope = np.full(len(old_slope), np.nan)
    for i in range(len(old_slope)):
        v = local_slope[similar[i]]
        if np.isfinite(v).any():
            sim_slope[i] = np.nanmedian(v)
    sim_slope = np.where(np.isfinite(sim_slope), sim_slope, old_slope)

    # Match the baseline model's target-specific trend strength rather than
    # quietly changing two things at once (neighbor method and multiplier).
    method = ns["selection"][target]
    if method.startswith("lag_mean10_a"):
        alpha = float(method.split("_a")[1])
    else:
        # CDD may select the nonlinear delta method. These are explicitly
        # new lag-mean candidates, not a claim to replace the delta formula.
        alpha = 0.5

    out = {
        "adaptive_trend": stats["mean10"][:, None] + alpha * adaptive[:, None] * gap,
        "similarity_neighbors": stats["mean10"][:, None] + alpha * sim_slope[:, None] * gap,
    }
    # Follow the same stale-city fallback and physical constraints as notebook.
    for name, f in out.items():
        f = ns["apply_fallback"](f, x, origin, stats["stale"])
        if target in ns["NONNEG"]:
            f = np.maximum(f, 0)
        out[name] = f
    return out


def gather_candidates(ns: dict, origins: list[int]) -> pd.DataFrame:
    rows = []
    for target in TREND_TARGETS:
        strict = np.where(ns["STRICT"][target], ns["A"][target], np.nan)
        for origin in origins:
            preds = experimental_forecasts(ns, target, origin)
            oi = ns["yidx"](origin)
            actual = strict[:, oi + 1:oi + 1 + HORIZON]
            ci, hi = np.nonzero(np.isfinite(actual))
            part = pd.DataFrame({"origin": origin, "target": target, "city": ci,
                                 "year": origin + hi + 1})
            for name, arr in preds.items():
                part[name] = arr[ci, hi]
            rows.append(part)
        print(f"Completed experimental forecasts: {target}")
    return pd.concat(rows, ignore_index=True)


def paired_evaluation(team: pd.DataFrame, candidate: pd.DataFrame, ns: dict,
                      folder: Path) -> None:
    paired = team.merge(candidate, on=KEY, how="inner", validate="one_to_one")
    cand_names = [c for c in candidate if c not in KEY]
    paired = paired.dropna(subset=["team_selected"] + cand_names)
    if paired.empty:
        raise RuntimeError("No matched forecasts; cannot compare candidates")
    print(f"\nPaired comparisons on {len(paired):,} observations")
    summary = []
    for target, g in paired.groupby("target"):
        for model in ["team_selected"] + cand_names:
            err = g[model] - g["actual"]
            summary.append({"target": target, "model": model, "n": len(g),
                            "mae": err.abs().mean(), "bias": err.mean()})
    result = pd.DataFrame(summary)
    result.to_csv(folder / "paired_candidate_metrics.csv", index=False)
    print(result.round(4).to_string(index=False))

    # Year-block bootstrap, keeping the same actual city-year pairs for all methods.
    rng = np.random.default_rng(42)
    bootstrap_rows = []
    for target, g in paired.groupby("target"):
        yearly = (g.assign(**{f"diff_{m}": (g[m]-g.actual).abs() - (g.team_selected-g.actual).abs()
                            for m in cand_names})
                  .groupby("year")[[f"diff_{m}" for m in cand_names]].agg(["sum", "count"]))
        years = yearly.index.to_numpy()
        draws = rng.integers(0, len(years), size=(1000, len(years)))
        for model in cand_names:
            sums = yearly[(f"diff_{model}", "sum")].to_numpy()
            counts = yearly[(f"diff_{model}", "count")].to_numpy()
            sampled = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
            bootstrap_rows.append({"target": target, "model": model,
                "delta_mae_vs_team": sums.sum() / counts.sum(),
                "ci95_low": np.quantile(sampled, .025),
                "ci95_high": np.quantile(sampled, .975),
                "p_lower_mae": (sampled < 0).mean()})
    boot = pd.DataFrame(bootstrap_rows)
    boot.to_csv(folder / "paired_candidate_bootstrap.csv", index=False)
    print("\nCandidate − team MAE (negative is better):")
    print(boot.round(4).to_string(index=False))
    paired.to_csv(folder / "paired_forecasts.csv", index=False)
    print("\nNOTE: Exploratory only. The baseline's model selection used overlapping historical folds;")
    print("do not claim these comparisons are independent holdout evidence.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostics-only", action="store_true")
    ap.add_argument("--quick", action="store_true", help="Only origins 2004 and 2011; for smoke testing, not selection")
    args = ap.parse_args()
    ns = mc.load_team_pipeline()
    assert ns["CUTOFF"] == 2018
    origins = [2004, 2011] if args.quick else [o for o in ns["ORIGINS"] if o >= 1990]
    RESULTS.mkdir(parents=True, exist_ok=True)
    team = selected_team_rows(ns, origins)
    diagnostics(team, ns, RESULTS)
    if not args.diagnostics_only:
        candidate = gather_candidates(ns, origins)
        paired_evaluation(team, candidate, ns, RESULTS)
    print(f"\nResults saved in: {RESULTS}")


if __name__ == "__main__":
    main()