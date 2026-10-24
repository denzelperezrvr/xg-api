import json
import pytest
from fastapi.testclient import TestClient

from xg.api import app

client = TestClient(app)

PENALTY_SPOT = {"x": 108, "y": 40, "shot_body_part": "Right Foot"}
MIDFIELD = {"x": 60, "y": 40, "shot_body_part": "Right Foot"}

def test_index_serves_ui():
    r = client.get("/")
    assert r.status_code == 200
    assert "Calculadora de xG" in r.text

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_probability():
    r = client.post("/predict", json=PENALTY_SPOT)
    assert r.status_code == 200
    assert 0 <= r.json()["xg"] <= 1


def test_closer_shot_has_higher_xg():
    near = client.post("/predict", json=PENALTY_SPOT).json()["xg"]
    far = client.post("/predict", json=MIDFIELD).json()["xg"]
    assert near > far


def test_header_is_worse_than_foot_from_same_spot():
    foot = client.post("/predict", json=PENALTY_SPOT).json()["xg"]
    head = client.post("/predict", json={**PENALTY_SPOT, "shot_body_part": "Head"}).json()["xg"]
    assert head < foot


@pytest.mark.parametrize(
    "payload",
    [
        {**PENALTY_SPOT, "x": 500},                      # fuera del campo
        {**PENALTY_SPOT, "y": -3},
        {**PENALTY_SPOT, "shot_body_part": "Hed"},       # categoría inválida
        {"x": 100, "y": 40},                             # falta la parte del cuerpo
    ],
)
def test_invalid_input_returns_422(payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_matches_single():
    single = client.post("/predict", json=PENALTY_SPOT).json()["xg"]
    r = client.post("/predict/batch", json={"shots": [PENALTY_SPOT, MIDFIELD]})
    assert r.status_code == 200
    xgs = r.json()["xg"]
    assert len(xgs) == 2
    assert xgs[0] == single


def test_empty_batch_returns_422():
    assert client.post("/predict/batch", json={"shots": []}).status_code == 422

def test_prediction_is_logged(tmp_path, monkeypatch):
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv("XG_LOG_PATH", str(log))
    client.post("/predict", json=PENALTY_SPOT)
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["x"] == 108
    assert 0 <= record["xg"] <= 1