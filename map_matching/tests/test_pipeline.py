import numpy as np
import pandas as pd

from map_matching.candidates import CandidateConfig, generate_candidates
from map_matching.features import FEATURE_COLUMNS, build_feature_table
from utils.geo import bearing_deg, point_segment_distance


class FakeTrip:
    trip_id = "fake"

    def __init__(self):
        self.track = pd.DataFrame({
            "obs_idx": [0, 1], "lon": [0.0, 0.001], "lat": [0.0, 0.0],
            "t": [0.0, 1.0], "x": [0.0, 10.0], "y": [0.0, 0.0],
        })
        self.arcs = pd.DataFrame({
            "arc_id": [0, 1, 2], "u": [0, 1, 2], "v": [1, 2, 3],
            "ax": [0.0, 0.0, 1000.0], "ay": [0.0, 100.0, 0.0],
            "bx": [100.0, 100.0, 1100.0], "by": [0.0, 100.0, 0.0],
            "length_m": [100.0, 100.0, 100.0],
        })


def test_point_segment_and_zero_length():
    distance, fraction = point_segment_distance(2.0, 3.0, 0.0, 0.0, 0.0, 0.0)
    assert distance ==  np.hypot(2.0, 3.0)
    assert fraction == 0.0


def test_candidates_radius_top_k_and_determinism():
    trip = FakeTrip()
    config = CandidateConfig(radius_m=15, top_k=1, cell_size_m=20)
    first = generate_candidates(trip, config)
    second = generate_candidates(trip, config)
    assert first.equals(second)
    assert first.arc_id.tolist() == [0, 0]
    assert first.candidate_rank.tolist() == [1, 1]


def test_feature_schema_and_no_label_leakage():
    labels = pd.DataFrame({"obs_idx": [0, 1], "true_arc_id": [0, 0]})
    table = build_feature_table(FakeTrip(), CandidateConfig(radius_m=15), labels)
    assert all(column in table for column in FEATURE_COLUMNS)
    assert table.label.tolist() == [1, 1]
    assert "true_arc_id" not in FEATURE_COLUMNS
    assert np.isfinite(table[FEATURE_COLUMNS].to_numpy()).all()


def test_bearing_is_compass_clockwise():
    assert bearing_deg(0, 1) == 0
    assert bearing_deg(1, 0) == 90
