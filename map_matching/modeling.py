"""Group-aware candidate scoring experiments and serializable inference."""
from dataclasses import dataclass
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
try:  # Keep CPU-only inference usable when the optional GPU experiment is absent.
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - covered by the declared production dependency
    XGBClassifier = None

from map_matching.features import FEATURE_COLUMNS


EXPERIMENTS = {
    "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=7)),
    "random_forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=7),
    "extra_trees": ExtraTreesClassifier(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=7),
    "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=7),
    "random_forest_balanced": RandomForestClassifier(
        n_estimators=150, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=7
    ),
}
if XGBClassifier is not None:
    EXPERIMENTS["xgboost_gpu"] = XGBClassifier(
        n_estimators=180, max_depth=6, learning_rate=0.08, subsample=0.9,
        colsample_bytree=0.9, eval_metric="logloss", tree_method="hist",
        device="cuda", random_state=7,
    )


def prepare_training_table(table):
    """Remove unlabelled observations while retaining a separate target column."""
    required = set(FEATURE_COLUMNS + ["label", "trip_id", "obs_idx"])
    missing = required.difference(table.columns)
    if missing:
        raise ValueError("missing columns: " + ", ".join(sorted(missing)))
    clean = table[table.label >= 0].copy()
    if clean.empty:
        raise ValueError("no labelled candidate rows available")
    return clean


def grouped_experiments(table, n_splits=5, verbose=True):
    """Evaluate six distinct models with trip-held-out cross-validation."""
    clean = prepare_training_table(table)
    x, y = clean[FEATURE_COLUMNS], clean.label
    groups = clean.trip_id
    splitter = GroupKFold(n_splits=min(n_splits, groups.nunique()))
    rows = []
    if verbose:
        print(
            f"[experiments] {len(EXPERIMENTS)} models | "
            f"{len(clean):,} labelled candidate rows | "
            f"{groups.nunique()} trips | {n_splits}-fold GroupKFold",
            flush=True,
        )
    fold_splits = list(splitter.split(x, y, groups))
    for name, estimator in EXPERIMENTS.items():
        start = time.perf_counter()
        scores = []
        if verbose:
            print(f"[model] START {name} ({type(estimator).__name__})", flush=True)
        for fold_number, (train_idx, valid_idx) in enumerate(fold_splits, 1):
            model = clone(estimator).fit(x.iloc[train_idx], y.iloc[train_idx])
            pred = _predict_for_model(model, x.iloc[valid_idx])
            fold_scores = {
                "accuracy": accuracy_score(y.iloc[valid_idx], pred),
                "macro_f1": f1_score(y.iloc[valid_idx], pred, average="macro", zero_division=0),
            }
            scores.append(fold_scores)
            if verbose:
                print(
                    f"[model] {name} fold={fold_number}/{len(fold_splits)} "
                    f"train_rows={len(train_idx):,} valid_rows={len(valid_idx):,} "
                    f"accuracy={fold_scores['accuracy']:.6f} "
                    f"macro_f1={fold_scores['macro_f1']:.6f}",
                    flush=True,
                )
        result = {
            "experiment": name, "model": type(estimator).__name__,
            "accuracy_mean": np.mean([s["accuracy"] for s in scores]),
            "accuracy_std": np.std([s["accuracy"] for s in scores]),
            "macro_f1_mean": np.mean([s["macro_f1"] for s in scores]),
            "macro_f1_std": np.std([s["macro_f1"] for s in scores]),
            "runtime_s": time.perf_counter() - start,
        }
        rows.append(result)
        if verbose:
            print(
                f"[model] DONE {name} runtime={result['runtime_s']:.2f}s "
                f"accuracy={result['accuracy_mean']:.6f}+/-{result['accuracy_std']:.6f} "
                f"macro_f1={result['macro_f1_mean']:.6f}+/-{result['macro_f1_std']:.6f}",
                flush=True,
            )
    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)


def _predict_for_model(model, x):
    """Run prediction safely across CPU/CUDA XGBoost boundaries."""
    if XGBClassifier is not None and isinstance(model, XGBClassifier):
        model.set_params(device="cpu")
        model.get_booster().set_param({"device": "cpu"})
    return model.predict(x)


def fit_final_model(table, output_path, experiment="xgboost_gpu"):
    """Fit the selected model on all permitted labelled rows."""
    clean = prepare_training_table(table)
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown experiment: {experiment}")
    model = clone(EXPERIMENTS[experiment]).fit(clean[FEATURE_COLUMNS], clean.label)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS, "classes": model.classes_.tolist()}, output_path)
    output_path.with_suffix(".json").write_text(json.dumps({"feature_columns": FEATURE_COLUMNS}, indent=2))
    return model


def fit_extra_trees_with_params(table, output_path, params):
    """Fit tuned Extra Trees as the final, inference-ready model."""
    clean = prepare_training_table(table)
    model = ExtraTreesClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=float(params["max_features"]),
        n_jobs=-1,
        random_state=7,
    ).fit(clean[FEATURE_COLUMNS], clean.label)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "classes": model.classes_.tolist(),
        "hyperparameters": params,
    }
    joblib.dump(bundle, output_path)
    output_path.with_suffix(".json").write_text(
        json.dumps({
            "model": "ExtraTreesClassifier",
            "feature_columns": FEATURE_COLUMNS,
            "hyperparameters": params,
            "classes": model.classes_.tolist(),
        }, indent=2)
    )
    return model


def nearest_candidate_baseline(table):
    """Evaluate the deterministic nearest-road candidate baseline."""
    clean = prepare_training_table(table)
    rank1 = clean[clean.candidate_rank == 1]
    if rank1.empty:
        raise ValueError("candidate rank 1 is missing")
    prediction = np.ones(len(rank1), dtype=int)
    label = rank1.label.to_numpy(dtype=int)
    return {
        "experiment": "nearest_road_baseline",
        "model": "rank1_nearest_candidate",
        "accuracy": float(accuracy_score(label, prediction)),
        "macro_f1": float(f1_score(label, prediction, average="macro", zero_division=0)),
        "n_observations": int(len(rank1)),
    }


def _cv_score(estimator, x, y, groups, n_splits):
    splitter = GroupKFold(n_splits=min(n_splits, groups.nunique()))
    scores = []
    for train_idx, valid_idx in splitter.split(x, y, groups):
        model = clone(estimator).fit(x.iloc[train_idx], y.iloc[train_idx])
        scores.append(f1_score(y.iloc[valid_idx], model.predict(x.iloc[valid_idx]),
                               average="macro", zero_division=0))
    return float(np.mean(scores))


def bayesian_optimize(table, n_trials=8, random_state=7, verbose=True):
    """Tune Extra Trees with a Gaussian-process surrogate and EI selection."""
    clean = prepare_training_table(table)
    x, y, groups = clean[FEATURE_COLUMNS], clean.label, clean.trip_id
    rng = np.random.default_rng(random_state)
    bounds = np.array([[50, 180], [2, 20], [1, 8], [0.4, 1.0]], dtype=float)
    points, scores, records = [], [], []
    for trial in range(n_trials):
        if len(points) < 3:
            point = rng.uniform(bounds[:, 0], bounds[:, 1])
        else:
            kernel = Matern(length_scale=np.ones(4), nu=2.5) + WhiteKernel(1e-5)
            surrogate = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=random_state)
            surrogate.fit(np.asarray(points), np.asarray(scores))
            pool = rng.uniform(bounds[:, 0], bounds[:, 1], size=(128, 4))
            mean, std = surrogate.predict(pool, return_std=True)
            incumbent = max(scores)
            improvement = mean - incumbent + 0.01 * std
            point = pool[int(np.argmax(improvement))]
        params = {
            "n_estimators": int(round(point[0])),
            "max_depth": int(round(point[1])),
            "min_samples_leaf": int(round(point[2])),
            "max_features": float(point[3]),
        }
        start = time.perf_counter()
        estimator = ExtraTreesClassifier(n_jobs=-1, random_state=random_state, **params)
        score = _cv_score(estimator, x, y, groups, n_splits=5)
        points.append(point)
        scores.append(score)
        records.append({"trial": trial, **params, "macro_f1": score,
                        "duration_s": time.perf_counter() - start})
        if verbose:
            print(
                f"[bayesian] trial={trial + 1}/{n_trials} "
                f"n_estimators={params['n_estimators']} max_depth={params['max_depth']} "
                f"min_samples_leaf={params['min_samples_leaf']} "
                f"max_features={params['max_features']:.6f} "
                f"macro_f1={score:.6f} duration={records[-1]['duration_s']:.2f}s",
                flush=True,
            )
    history = pd.DataFrame(records)
    return history, history.loc[history.macro_f1.idxmax()].to_dict()


@dataclass
class CandidateScorer:
    """Inference wrapper returning candidate IDs, scores, and predictions."""

    bundle: dict

    @classmethod
    def load(cls, path):
        return cls(joblib.load(path))

    def predict(self, feature_table):
        missing = set(self.bundle["feature_columns"]).difference(feature_table.columns)
        if missing:
            raise ValueError("missing feature columns: " + ", ".join(sorted(missing)))
        x = feature_table[self.bundle["feature_columns"]]
        model = self.bundle["model"]
        if XGBClassifier is not None and isinstance(model, XGBClassifier):
            model.set_params(device="cpu")
            model.get_booster().set_param({"device": "cpu"})
        score = model.predict_proba(x)[:, -1]
        output = feature_table[["trip_id", "obs_idx", "arc_id", "candidate_rank"]].copy()
        output["score"] = score
        output["prediction"] = (score >= 0.5).astype("int8")
        return output
