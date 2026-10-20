from importlib.resources import files

from fastapi import FastAPI, HTTPException, Query

from metafeed.recommender import (
    RecommendationEngine,
    load_dataset,
)
from metafeed.schemas import RecommendationResponse


dataset_path = files("metafeed").joinpath(
    "data", "sample.json"
)

engine = RecommendationEngine(
    load_dataset(dataset_path)
)

app = FastAPI(
    title="MetaFeed AI",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get(
    "/v1/users/{user_id}/recommendations",
    response_model=RecommendationResponse,
)
def get_recommendations(
    user_id: str,
    limit: int = Query(default=10, ge=1, le=100),
):
    try:
        results = engine.recommend(user_id, limit)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        ) from None

    return RecommendationResponse(
        user_id=user_id,
        algorithm="popularity_v1",
        recommendations=results,
    )
