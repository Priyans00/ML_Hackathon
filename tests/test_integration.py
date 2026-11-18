from backend.service import PathSnapService


def test_real_student_trip_end_to_end_contract():
    service = PathSnapService(evaluator_split="student_test")
    trip_ids = service.trip_ids()
    assert trip_ids
    result = service.match(trip_ids[0])
    assert result["trip_id"] == trip_ids[0]
    assert isinstance(result["gps"], list)
    assert isinstance(result["predicted_route"], list)
    forbidden = {"true_arc_id", "route", "route_idx", "label"}
    assert forbidden.isdisjoint(result)
