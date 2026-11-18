"""Run six group-aware experiments and save a reproducible final model."""
from pathlib import Path

import pandas as pd

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from map_matching.modeling import (
    bayesian_optimize,
    fit_extra_trees_with_params,
    grouped_experiments,
    nearest_candidate_baseline,
)


def main():
    table_path = Path("results/phase2/candidates.parquet")
    out = Path("results/phase3")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[phase3] loading {table_path}", flush=True)
    table = pd.read_parquet(table_path)
    print(
        f"[phase3] loaded rows={len(table):,} trips={table.trip_id.nunique()} "
        f"columns={len(table.columns)}",
        flush=True,
    )
    print("[phase3] starting six model-by-model grouped experiments", flush=True)
    comparison = grouped_experiments(table)
    comparison.to_csv(out / "experiment_comparison.csv", index=False)
    print(f"[phase3] experiments complete; writing comparison to {out / 'experiment_comparison.csv'}", flush=True)
    print("[phase3] starting Bayesian optimization", flush=True)
    trials, best = bayesian_optimize(table)
    trials.to_csv(out / "bayesian_trials.csv", index=False)
    (out / "best_parameters.json").write_text(pd.Series(best).to_json(indent=2))
    print(
        f"[phase3] Bayesian complete; best trial macro_f1={best['macro_f1']:.6f}",
        flush=True,
    )
    print("[phase3] retraining and saving tuned Extra Trees final model", flush=True)
    untuned_macro_f1 = float(comparison.loc[comparison.experiment == "extra_trees", "macro_f1_mean"].iloc[0])
    tuned_params = {
        "n_estimators": int(best["n_estimators"]),
        "max_depth": int(best["max_depth"]),
        "min_samples_leaf": int(best["min_samples_leaf"]),
        "max_features": float(best["max_features"]),
    }
    fit_extra_trees_with_params(table, out / "final_model.joblib", tuned_params)
    tune_comparison = pd.DataFrame([
        {"configuration": "untuned_extra_trees", "macro_f1": untuned_macro_f1},
        {"configuration": "bayesian_tuned_extra_trees", "macro_f1": float(best["macro_f1"])},
    ])
    tune_comparison.to_csv(out / "tuned_vs_untuned.csv", index=False)
    baseline = nearest_candidate_baseline(table)
    pd.DataFrame([baseline]).to_csv(out / "baseline.csv", index=False)
    print(f"[phase3] saved final model to {out / 'final_model.joblib'}", flush=True)
    print(comparison.to_string(index=False), flush=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = comparison.sort_values("macro_f1_mean")
    ax.barh(ordered.experiment, ordered.macro_f1_mean,
            xerr=ordered.macro_f1_std, color="#4c78a8")
    ax.set(title="Grouped-CV macro-F1 by experiment", xlabel="macro-F1")
    fig.tight_layout()
    fig.savefig(out / "experiment_comparison.png", dpi=150)
    plt.close(fig)

    trial_plot = pd.read_csv(out / "bayesian_trials.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(trial_plot.trial, trial_plot.macro_f1, marker="o")
    ax.set(title="Bayesian optimization history", xlabel="trial", ylabel="macro-F1")
    fig.tight_layout()
    fig.savefig(out / "bayesian_history.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
