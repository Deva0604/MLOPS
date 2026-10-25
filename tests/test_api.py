
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200

    result = response.json()
    assert result["target"] == "dynamic_risk_category"
    assert set(result["classes"]) == {"LOW", "MEDIUM", "HIGH", "CLOSED"}


def test_prediction_endpoint(client):
    payload = {
        "road_length_m": 500,
        "current_speed_kmh": 25,
        "free_flow_speed_kmh": 50,
        "congestion_ratio": 0.5,
        "temperature_c": 32,
        "humidity_percent": 70,
        "precipitation_mm": 0,
        "wind_speed_kmh": 12,
        "road_closure": False,
        "highway": "['residential']",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    result = response.json()
    assert result["predicted_risk_category"] in {
        "LOW", "MEDIUM", "HIGH", "CLOSED"
    }

    probabilities = result["class_probabilities"]
    assert set(probabilities) == {"LOW", "MEDIUM", "HIGH", "CLOSED"}
    assert all(0 <= p <= 1 for p in probabilities.values())
    assert abs(sum(probabilities.values()) - 1.0) < 0.01


def test_prediction_rejects_invalid_humidity(client):
    payload = {
        "road_length_m": 500,
        "current_speed_kmh": 25,
        "free_flow_speed_kmh": 50,
        "congestion_ratio": 0.5,
        "temperature_c": 32,
        "humidity_percent": 150,
        "precipitation_mm": 0,
        "wind_speed_kmh": 12,
        "road_closure": False,
        "highway": "['residential']",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
