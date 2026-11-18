import pandas as pd
import pytest
import tempfile
from pathlib import Path

import numpy as np
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from map_matching.features import FEATURE_COLUMNS
from map_matching.modeling import CandidateScorer, nearest_candidate_baseline, prepare_training_table


def test_training_rejects_missing_schema():
    with pytest.raises(ValueError, match="missing columns"):
        prepare_training_table(pd.DataFrame({"label": [1]}))


def test_training_keeps_unlabelled_rows_out():
    rows = {column: [0.0, 0.0] for column in FEATURE_COLUMNS}
    rows.update({"trip_id": ["a", "a"], "obs_idx": [0, 1], "label": [1, -1]})
    clean = prepare_training_table(pd.DataFrame(rows))
    assert clean.obs_idx.tolist() == [0]


def test_xgboost_cuda_model_can_be_loaded_and_predicted_on_cpu():
    if XGBClassifier is None:
        pytest.skip("xgboost is not installed in this test environment")
    x = np.random.default_rng(7).random((100, len(FEATURE_COLUMNS)))
    y = (x[:, 0] > 0.5).astype(int)
    model = XGBClassifier(n_estimators=5, max_depth=2, device="cuda",
                          tree_method="hist", random_state=7)
    model.fit(x, y)
    frame = pd.DataFrame(x, columns=FEATURE_COLUMNS)
    frame.insert(0, "trip_id", "test")
    frame.insert(1, "obs_idx", np.arange(len(frame)))
    frame["arc_id"] = 1
    frame["candidate_rank"] = 1
    bundle = {"model": model, "feature_columns": FEATURE_COLUMNS,
              "classes": model.classes_.tolist()}
    scorer = CandidateScorer(bundle)
    result = scorer.predict(frame)
    assert len(result) == len(frame)
    assert {"trip_id", "obs_idx", "arc_id", "candidate_rank", "score", "prediction"} <= set(result)


def test_nearest_candidate_baseline_is_predictive():
    rows = {column: [0.0, 0.0] for column in FEATURE_COLUMNS}
    rows.update({
        "trip_id": ["a", "a"],
        "obs_idx": [0, 1],
        "candidate_rank": [1, 2],
        "label": [1, 0],
    })
    baseline = nearest_candidate_baseline(pd.DataFrame(rows))
    assert baseline["accuracy"] == 1.0
