"""Evaluator-safe FastAPI application for Laptop 1."""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.service import PathSnapService, UnknownTripError


class MatchRequest(BaseModel):
    trip_id: str = Field(pattern=r"^\d{8}$")


def create_app(service: PathSnapService | None = None) -> FastAPI:
    logging.basicConfig(level=os.getenv("PATHSNAP_LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    evaluator_split = os.getenv("PATHSNAP_EVALUATOR_SPLIT", "student_test")
    # Importing this module must not deserialize the model. In particular, API
    # tests can inject a small fake service and Uvicorn can finish starting
    # before the first inference request. The real service is cached per app.
    loaded_service = service

    def get_service():
        nonlocal loaded_service
        if loaded_service is None:
            loaded_service = PathSnapService(evaluator_split=evaluator_split)
        return loaded_service
    app = FastAPI(title="PathSnap API", version="1.0.0")
    origins = [origin.strip() for origin in os.getenv("PATHSNAP_CORS_ORIGINS", "*").split(",")]
    app.add_middleware(
        CORSMiddleware, allow_origins=origins, allow_credentials=False,
        allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
    )

    @app.get("/health")
    async def health():
        active_service = get_service()
        return {"status": "ok", "evaluator_split": active_service.evaluator_split,
                "available_trips": len(active_service.trip_ids())}

    @app.get("/api/trips")
    async def trips():
        return {"trips": [{"trip_id": trip_id} for trip_id in get_service().trip_ids()]}

    @app.get("/api/trips/{trip_id}")
    async def trip_data(trip_id: str):
        try:
            return get_service().load_observations(trip_id)
        except UnknownTripError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/match")
    async def match(request: MatchRequest):
        try:
            return get_service().match(request.trip_id)
        except UnknownTripError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            logging.getLogger(__name__).exception("invalid matching request for trip=%s", request.trip_id)
            raise HTTPException(status_code=422, detail=f"unable to process trip: {exc}") from exc
        except Exception as exc:  # Do not leak paths, model internals, or labels to the UI.
            logging.getLogger(__name__).exception("matching failed for trip=%s", request.trip_id)
            raise HTTPException(status_code=500, detail="matching failed; check Laptop 1 API logs") from exc

    return app


app = create_app()
