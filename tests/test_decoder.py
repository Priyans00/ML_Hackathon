import pandas as pd

from map_matching.decoder import DecoderConfig, ViterbiDecoder


class FakeTrip:
    trip_id = "fake"

    def __init__(self):
        self.arcs = pd.DataFrame({
            "arc_id": [0, 1, 2], "u": [0, 1, 3], "v": [1, 2, 4],
            "length_m": [10.0, 10.0, 10.0],
        })


def _rows(values):
    return pd.DataFrame(values, columns=[
        "obs_idx", "arc_id", "projection_fraction", "score", "t", "x", "y",
    ])


def test_viterbi_prefers_a_feasible_sequence_over_independent_peak():
    # Arc 2 is disconnected. Its first-observation probability is highest, but
    # only 0 -> 1 gives a feasible two-observation route.
    candidates = _rows([
        (0, 0, 0.2, 0.60, 0.0, 2.0, 0.0),
        (0, 2, 0.2, 0.99, 0.0, 2.0, 0.0),
        (1, 1, 0.2, 0.90, 1.0, 12.0, 0.0),
    ])
    decoded = ViterbiDecoder(FakeTrip()).decode(candidates)
    assert decoded.arc_id.tolist() == [0, 1]
    assert decoded.segment_id.tolist() == [0, 0]


def test_missing_observation_candidates_start_a_new_segment():
    candidates = _rows([
        (0, 0, 0.1, 0.8, 0.0, 1.0, 0.0),
        (2, 1, 0.1, 0.8, 2.0, 11.0, 0.0),
    ])
    decoded = ViterbiDecoder(FakeTrip()).decode(candidates)
    assert decoded.obs_idx.tolist() == [0, 2]
    assert decoded.segment_id.tolist() == [0, 1]


def test_large_gps_gap_restarts_without_dropping_predictions():
    candidates = _rows([
        (0, 0, 0.1, 0.8, 0.0, 1.0, 0.0),
        (1, 1, 0.1, 0.8, 90.0, 11.0, 0.0),
    ])
    decoded = ViterbiDecoder(FakeTrip(), DecoderConfig(reset_gap_s=60)).decode(candidates)
    assert decoded.arc_id.tolist() == [0, 1]
    assert decoded.segment_id.tolist() == [0, 1]
