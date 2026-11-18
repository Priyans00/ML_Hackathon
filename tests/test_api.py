import asyncio

import httpx

from backend.main import create_app
from backend.service import UnknownTripError


class FakeService:
    evaluator_split = "student_test"

    def trip_ids(self):
        return ["00000000"]

    def load_observations(self, trip_id):
        if trip_id != "00000000":
            raise UnknownTripError("not available")
        return {"trip_id": trip_id, "n_observations": 1,
                "gps": [{"obs_idx": 0, "t": 0.0, "lon": 1.0, "lat": 2.0}]}

    def match(self, trip_id):
        if trip_id != "00000000":
            raise UnknownTripError("not available")
        return {"trip_id": trip_id, "gps": [], "predicted_route": [],
                "diagnostics": {"n_observations": 1, "n_candidates": 1, "n_decoded": 1, "n_segments": 1}}


def test_api_lists_only_configured_trips_and_never_returns_labels():
    async def exercise_api():
        transport = httpx.ASGITransport(app=create_app(FakeService()))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/health")).status_code == 200
            assert (await client.get("/api/trips")).json() == {"trips": [{"trip_id": "00000000"}]}
            response = await client.post("/api/match", json={"trip_id": "00000000"})
            assert response.status_code == 200
            assert "true_arc_id" not in response.text
            assert (await client.post("/api/match", json={"trip_id": "00000001"})).status_code == 404

    asyncio.run(exercise_api())
