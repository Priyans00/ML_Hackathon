"""Server-side inference orchestration.

This service deliberately never reads ``Trip.route`` while making a
prediction.  It only loads the network, raw observations, candidate generator,
feature builder, trained scorer, and decoder required for inference.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from map_matching.candidates import CandidateConfig
from map_matching.decoder import DecoderConfig, ViterbiDecoder
from map_matching.features import build_feature_table
from map_matching.modeling import CandidateScorer
from utils.data_loader import DATA_ROOT, load_trip


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "results" / "phase3" / "final_model.joblib"


class UnknownTripError(ValueError):
    """Raised when a request is outside the configured evaluator partition."""


class PathSnapService:
    """Own model and dataset resources used by the API process on Laptop 1."""

    def __init__(self, *, model_path: Path | str = DEFAULT_MODEL_PATH,
                 data_root: Path | str = DATA_ROOT, evaluator_split: str = "student_test",
                 scorer=None, candidate_config: CandidateConfig | None = None,
                 decoder_config: DecoderConfig | None = None):
        self.model_path = Path(model_path)
        self.data_root = Path(data_root)
        self.evaluator_split = evaluator_split
        self.scorer = scorer or CandidateScorer.load(self.model_path)
        self.candidate_config = candidate_config or CandidateConfig()
        self.decoder_config = decoder_config or DecoderConfig()

    @property
    def _split_dir(self) -> Path:
        return self.data_root / self.evaluator_split

    def trip_ids(self) -> list[str]:
        if not self._split_dir.is_dir():
            return []
        return [path.name for path in sorted(self._split_dir.iterdir()) if path.is_dir()]

    def _require_trip(self, trip_id: str) -> None:
        if trip_id not in self.trip_ids():
            raise UnknownTripError(f"trip {trip_id!r} is not available for evaluator inference")

    def load_observations(self, trip_id: str) -> dict:
        self._require_trip(trip_id)
        trip = load_trip(trip_id, self.evaluator_split, root=self.data_root)
        return {
            "trip_id": trip.trip_id,
            "n_observations": int(len(trip.track)),
            "gps": [
                {"obs_idx": int(row.obs_idx), "t": float(row.t),
                 "lon": float(row.lon), "lat": float(row.lat)}
                for row in trip.track.itertuples(index=False)
            ],
        }

    @staticmethod
    def _route_payload(trip, decoded: pd.DataFrame) -> list[dict]:
        if decoded.empty:
            return []
        node_lookup = trip.nodes.set_index("node_id")
        arc_lookup = trip.arcs.set_index("arc_id")
        rows = []
        for row in decoded.itertuples(index=False):
            arc = arc_lookup.loc[int(row.arc_id)]
            start, end = node_lookup.loc[int(arc.u)], node_lookup.loc[int(arc.v)]
            rows.append({
                "obs_idx": int(row.obs_idx),
                "arc_id": int(row.arc_id),
                "confidence": round(float(row.score), 6),
                "segment_id": int(row.segment_id),
                "from": {"lon": float(start.lon), "lat": float(start.lat)},
                "to": {"lon": float(end.lon), "lat": float(end.lat)},
            })
        return rows

    def match(self, trip_id: str) -> dict:
        """Run the label-free candidate, scoring, and HMM pipeline for one trip."""
        self._require_trip(trip_id)
        trip = load_trip(trip_id, self.evaluator_split, root=self.data_root)
        features = build_feature_table(trip, config=self.candidate_config)
        if features.empty:
            LOGGER.warning("trip=%s had no candidates", trip_id)
            return {
                "trip_id": trip_id, "gps": self.load_observations(trip_id)["gps"],
                "predicted_route": [],
                "diagnostics": {"n_observations": int(len(trip.track)), "n_candidates": 0,
                                "n_decoded": 0, "n_segments": 0},
            }
        scored = self.scorer.predict(features)
        decoder_input = features.merge(
            scored[["obs_idx", "arc_id", "score"]], on=["obs_idx", "arc_id"],
            how="inner", validate="one_to_one",
        )
        decoded = ViterbiDecoder(trip, self.decoder_config).decode(decoder_input)
        LOGGER.info("trip=%s observations=%d candidates=%d decoded=%d segments=%d",
                    trip_id, len(trip.track), len(features), len(decoded),
                    decoded.segment_id.nunique() if not decoded.empty else 0)
        return {
            "trip_id": trip_id,
            "gps": [
                {"obs_idx": int(row.obs_idx), "t": float(row.t),
                 "lon": float(row.lon), "lat": float(row.lat)}
                for row in trip.track.itertuples(index=False)
            ],
            "predicted_route": self._route_payload(trip, decoded),
            "diagnostics": {
                "n_observations": int(len(trip.track)),
                "n_candidates": int(len(features)),
                "n_decoded": int(len(decoded)),
                "n_segments": int(decoded.segment_id.nunique()) if not decoded.empty else 0,
            },
        }
