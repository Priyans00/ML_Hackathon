"""Finalize the tuned Extra Trees model without rerunning experiments."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from map_matching.modeling import (
    fit_extra_trees_with_params,
    nearest_candidate_baseline,
)


def main():
    table = pd.read_parquet("results/phase2/candidates.parquet")
    out = Path("results/phase3")
    out.mkdir(parents=True, exist_ok=True)
    best = json.loads((out / "best_parameters.json").read_text())
    params = {
        "n_estimators": int(best["n_estimators"]),
        "max_depth": int(best["max_depth"]),
        "min_samples_leaf": int(best["min_samples_leaf"]),
        "max_features": float(best["max_features"]),
    }
    fit_extra_trees_with_params(table, out / "final_model.joblib", params)

    comparison = pd.read_csv(out / "experiment_comparison.csv")
    untuned = float(comparison.loc[comparison.experiment == "extra_trees", "macro_f1_mean"].iloc[0])
    pd.DataFrame([
        {"configuration": "untuned_extra_trees", "macro_f1": untuned},
        {"configuration": "bayesian_tuned_extra_trees", "macro_f1": float(best["macro_f1"])},
    ]).to_csv(out / "tuned_vs_untuned.csv", index=False)
    pd.DataFrame([nearest_candidate_baseline(table)]).to_csv(out / "baseline.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = comparison.sort_values("macro_f1_mean")
    ax.barh(ordered.experiment, ordered.macro_f1_mean, xerr=ordered.macro_f1_std, color="#4c78a8")
    ax.set(title="Grouped-CV macro-F1 by experiment", xlabel="macro-F1")
    fig.tight_layout()
    fig.savefig(out / "experiment_comparison.png", dpi=150)
    plt.close(fig)

    trials = pd.read_csv(out / "bayesian_trials.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(trials.trial, trials.macro_f1, marker="o")
    ax.set(title="Bayesian optimization history", xlabel="trial", ylabel="macro-F1")
    fig.tight_layout()
    fig.savefig(out / "bayesian_history.png", dpi=150)
    plt.close(fig)
    print("finalized", out / "final_model.joblib")


if __name__ == "__main__":
    main()
