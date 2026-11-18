"""Compare independent classifier choices and HMM-decoded paths safely.

Use only ``train`` or ``student_test`` development data. This script refuses
an evaluator partition so held-out evaluator data can never affect decoder
configuration or model selection.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_processing.label_alignment import label_trip
from map_matching.candidates import CandidateConfig
from map_matching.decoder import DecoderConfig, ViterbiDecoder, continuity_metrics
from map_matching.features import build_feature_table
from map_matching.modeling import CandidateScorer
from utils.data_loader import DATA_ROOT, list_trips, load_trip


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "results" / "phase3" / "final_model.joblib"


def _independent_top1(decoder_input: pd.DataFrame) -> pd.DataFrame:
    chosen = decoder_input.sort_values(
        ["obs_idx", "score", "candidate_rank", "arc_id"],
        ascending=[True, False, True, True], kind="mergesort",
    ).groupby("obs_idx", as_index=False).first()
    return chosen[["obs_idx", "arc_id", "score"]].assign(segment_id=0, decoder_reset=False)


def _arc_accuracy(selected: pd.DataFrame, labels: pd.DataFrame) -> tuple[float, int]:
    compared = selected.merge(labels[["obs_idx", "true_arc_id"]], on="obs_idx", how="inner")
    compared = compared[compared.true_arc_id >= 0]
    if compared.empty:
        return np.nan, 0
    return float((compared.arc_id == compared.true_arc_id).mean()), int(len(compared))


def evaluate_trip(trip, scorer, candidate_config=None, decoder_config=None) -> dict:
    """Evaluate one permitted development trip; inference itself uses no labels."""
    candidate_config = candidate_config or CandidateConfig()
    decoder_config = decoder_config or DecoderConfig()
    features = build_feature_table(trip, config=candidate_config)
    if features.empty:
        return {"trip_id": trip.trip_id, "n_candidates": 0, "n_decoded": 0}
    scored = scorer.predict(features)
    decoder_input = features.merge(
        scored[["obs_idx", "arc_id", "score"]], on=["obs_idx", "arc_id"],
        how="inner", validate="one_to_one",
    )
    classifier = _independent_top1(decoder_input)
    decoded = ViterbiDecoder(trip, decoder_config).decode(decoder_input)
    labels, _ = label_trip(trip)  # Offline development evaluation only.
    classifier_accuracy, n_evaluated = _arc_accuracy(classifier, labels)
    decoded_accuracy, _ = _arc_accuracy(decoded, labels)
    classifier_continuity = continuity_metrics(classifier, decoder_input, trip, decoder_config)
    decoded_continuity = continuity_metrics(decoded, decoder_input, trip, decoder_config)
    return {
        "trip_id": trip.trip_id,
        "n_observations": int(len(trip.track)),
        "n_candidates": int(len(features)),
        "n_evaluated_observations": n_evaluated,
        "n_decoded": int(len(decoded)),
        "classifier_arc_accuracy": classifier_accuracy,
        "decoded_arc_accuracy": decoded_accuracy,
        "classifier_plausible_transition_rate": classifier_continuity["plausible_transition_rate"],
        "decoded_plausible_transition_rate": decoded_continuity["plausible_transition_rate"],
        "classifier_pairs": classifier_continuity["n_pairs"],
        "decoded_pairs": decoded_continuity["n_pairs"],
        "decoded_segments": decoded_continuity["n_segments"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "student_test"), default="student_test")
    parser.add_argument("--trip", dest="trip_id", help="optional one-trip development check")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "phase4")
    args = parser.parse_args()

    scorer = CandidateScorer.load(args.model)
    trip_ids = [args.trip_id] if args.trip_id else [trip_id for split, trip_id in list_trips((args.split,), DATA_ROOT)]
    rows = []
    for trip_id in trip_ids:
        trip = load_trip(trip_id, args.split, DATA_ROOT)
        result = evaluate_trip(trip, scorer)
        rows.append(result)
        print(f"{args.split} {trip_id}: decoded_accuracy={result.get('decoded_arc_accuracy', np.nan):.6f} "
              f"continuity={result.get('decoded_plausible_transition_rate', np.nan):.6f}", flush=True)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    per_trip = pd.DataFrame(rows)
    per_trip.to_csv(output / f"{args.split}_per_trip.csv", index=False)
    numeric = per_trip.select_dtypes(include="number").mean(numeric_only=True).to_frame("mean").T
    numeric.insert(0, "split", args.split)
    numeric.to_csv(output / f"{args.split}_comparison.csv", index=False)
    print(f"wrote {output / f'{args.split}_per_trip.csv'}")


if __name__ == "__main__":
    main()
