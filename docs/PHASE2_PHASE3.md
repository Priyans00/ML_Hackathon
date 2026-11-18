# Phase 2 and Phase 3 interfaces

## Phase 2

`map_matching.candidates.generate_candidates(trip, CandidateConfig(...))`
uses a uniform metre-grid over valid arc bounding boxes, followed by exact
point-to-segment distance. Results are deterministic and keyed by
`(trip_id, obs_idx, arc_id)`. The default search is 75 m and top-8 candidates.

`map_matching.features.build_feature_table` adds distance, projection fraction,
arc length, road bearing, GPS heading, heading error, displacement, elapsed
time, and speed. `FEATURE_COLUMNS` is the authoritative stable feature order.
Labels are optional and are kept in the separate `label` column; unlabelled
observations have `label=-1` and are excluded by the training helper.

Run candidate generation and coverage reporting with:

```text
E:\anaconda\envs\lab_env\python.exe -m scripts.run_phase2
```

It writes `results/phase2/candidates.parquet` and `coverage.csv`.

## Phase 3

`map_matching.modeling.grouped_experiments` evaluates six distinct model
families/configurations with `GroupKFold` by trip ID. The XGBoost experiment
uses the verified CUDA device in `lab_env`. `fit_final_model` saves a joblib
bundle containing the estimator and feature ordering. Load it with
`CandidateScorer.load(path)` and call `predict(feature_table)`; the result
contains trip ID, observation ID, arc ID, rank, score, and binary prediction.

Run experiments after Phase 2:

```text
E:\anaconda\envs\lab_env\python.exe -m scripts.run_phase3
```

Ground-truth labels are used only to construct training targets. They are not
included in `FEATURE_COLUMNS` and are not required at inference time.

The selected final model is the tuned Extra Trees configuration:
`n_estimators=152`, `max_depth=10`, `min_samples_leaf=4`, and
`max_features=0.6923585529`. Bayesian trial 7 produced macro-F1 0.865597,
compared with 0.860784 for untuned Extra Trees. Final inference scores the
candidate table and returns candidate IDs, ranks, probabilities, and
predictions.
