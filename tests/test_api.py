from fastapi.testclient import TestClient

from metafeed.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommendation_endpoint():
    response = client.get(
        "/v1/users/alice/recommendations"
    )
    assert response.status_code == 200

    body = response.json()
    assert body["algorithm"] == "popularity_v1"
    assert body["user_id"] == "alice"
    assert [
        item["item_id"]
        for item in body["recommendations"]
    ] == [
        "video_003",
        "video_004",
        "video_005",
        "video_006",
    ]


def test_unknown_user_returns_404():
    response = client.get(
        "/v1/users/unknown/recommendations"
    )
    assert response.status_code == 404


def test_invalid_limits_return_422():
    for value in ("0", "-1", "101", "abc", "1.5"):
        response = client.get(
            "/v1/users/alice/recommendations",
            params={"limit": value},
        )
        assert response.status_code == 422


def test_limit_and_catalog_exhaustion():
    response = client.get(
        "/v1/users/alice/recommendations",
        params={"limit": 100},
    )
    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 4
