"""Generate a label-free Folium Phase 4 predicted-route map."""
import argparse
from pathlib import Path

from backend.service import PathSnapService
from eda.plot_trip import plot_trip
from utils.data_loader import load_trip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trip", default="00000052")
    parser.add_argument("--split", default="student_test")
    parser.add_argument("--output", type=Path,
                        default=Path("results/phase4/predicted_trip_00000052.html"))
    args = parser.parse_args()
    service = PathSnapService(evaluator_split=args.split)
    trip = load_trip(args.trip, args.split)
    result = service.match(args.trip)
    predicted = [int(segment["arc_id"]) for segment in result["predicted_route"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_trip(args.trip, args.split, trip=trip, predicted_arcs=predicted).save(args.output)
    print(f"saved {args.output} with {len(predicted)} predicted arc segments")


if __name__ == "__main__":
    main()
