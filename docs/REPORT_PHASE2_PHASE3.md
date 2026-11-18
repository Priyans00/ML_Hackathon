# PathSnap Phase 2 and Phase 3 Report

## Phase 2

### Candidate pipeline

Candidate generation uses a per-trip uniform metre-grid spatial index. The grid
is built once for each road network and exact point-to-segment distance is
calculated only for bounding-box-filtered arc candidates. The default
configuration is a 75 m radius and top-8 candidates. Candidate IDs are the
original per-trip arc IDs. Invalid coordinates and zero-length segments are
handled safely.

### Features

The stable model feature order is candidate rank, GPS-to-road distance,
projection fraction, arc length, candidate bearing, GPS heading, heading error,
GPS displacement, elapsed time, and GPS speed. Labels remain in a separate
`label` column and are never in `FEATURE_COLUMNS`.

### Coverage and label quality

The full training benchmark covered 80 trips. Overall observation coverage was
93.4708% and true-ground-truth candidate recall was 97.4527%. Trip 33 had
100.0% observation coverage and 80.1253% candidate recall. Trip 54 had 33.2470%
observation coverage and 82.9291% candidate recall.

The label alignment regeneration confirmed the preliminary audit findings:
Trip 33 was 59.0178% unaligned and Trip 54 was 88.8987% unaligned. These trips
were not deleted or relabelled. Unlabelled observations were isolated with
`label=-1` and excluded only from the supervised training table.

### Runtime

The full Phase 2 run took 565.1 seconds. Loading, label alignment, and
candidate generation/feature extraction were separately timed. Candidate
generation/feature extraction was the dominant stage. The representative
benchmark trip took 6.29 seconds after spatial-index optimization.

## Phase 3

### Formulation

The problem is binary candidate classification: a candidate is either the
reference-road candidate (`label=1`) or not (`label=0`). Macro-F1 and accuracy
are reported because candidate classes are imbalanced and one-vs-rest candidate
selection is not a route reconstruction metric.

### Validation

All candidates for one trip remain in one fold. `GroupKFold` with 80 training
trips and five folds prevents same-trip candidate leakage. Student testing
ground truth was not used for training or tuning.

### Baseline and experiments

The nearest-road rank-1 baseline produced accuracy 0.552895 and macro-F1
0.356041.

| Experiment | Accuracy mean | Accuracy std | Macro-F1 mean | Macro-F1 std |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.921783 | 0.015229 | 0.844791 | 0.025860 |
| Random Forest | 0.928242 | 0.020234 | 0.859179 | 0.037671 |
| Extra Trees | 0.929018 | 0.019480 | 0.860784 | 0.035349 |
| HistGradientBoosting | 0.928350 | 0.019791 | 0.859815 | 0.037274 |
| Balanced Random Forest | 0.928056 | 0.020350 | 0.862565 | 0.036660 |
| CUDA XGBoost | 0.928419 | 0.019939 | 0.860063 | 0.037400 |

XGBoost CUDA training succeeded on the RTX 3050 and its five-fold run took
11.36 seconds. XGBoost prediction was explicitly switched to CPU to avoid
CPU/NumPy-to-CUDA mismatched input behavior. Candidate-generation and model
metrics do not represent final route reconstruction accuracy.

### Bayesian optimization

A Gaussian-process Bayesian search tuned Extra Trees over eight trials.
The objective was five-fold trip-grouped macro-F1. The best trial was trial 7
with `n_estimators=152`, `max_depth=10`, `min_samples_leaf=4`, and
`max_features=0.6923585529`, yielding macro-F1 0.865597. Untuned Extra Trees
yielded 0.860784, for an observed +0.004813 improvement.

### Final artifact

The final model is tuned Extra Trees in `results/phase3/final_model.joblib`.
The loading and feature schema are described in `results/phase3/final_model.json`.

```python
from map_matching.modeling import CandidateScorer

scorer = CandidateScorer.load("results/phase3/final_model.joblib")
result = scorer.predict(candidate_feature_table)
```

`result` contains `trip_id`, `obs_idx`, `arc_id`, `candidate_rank`, `score`,
and `prediction`.
