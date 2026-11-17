from fastapi.testclient import TestClient

from src.food11 import serve


class FakeModel:
    """Stands in for the MLflow model: one row of 11 scores, Bread highest."""

    def predict(self, x):
        assert x.shape == (1, 3, 128, 128)  # the API sends one preprocessed image
        return [[0.90] + [0.01] * 10]


def test_health(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as client:
        assert client.get("/health").status_code == 200


def test_predict_returns_valid_json(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as client:
        monkeypatch.setattr(serve, "model", FakeModel())
        with open("tests/fixtures/bread.jpg", "rb") as f:
            r = client.post("/predict", files={"file": ("bread.jpg", f, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["category"] == "Bread"
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_without_model_returns_503(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    monkeypatch.setattr(serve, "model", None)
    with TestClient(serve.app) as client, open("tests/fixtures/bread.jpg", "rb") as f:
        r = client.post("/predict", files={"file": ("bread.jpg", f, "image/jpeg")})
    assert r.status_code == 503


def test_predict_rejects_non_image(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as client:
        monkeypatch.setattr(serve, "model", FakeModel())
        r = client.post("/predict", files={"file": ("notes.txt", b"not an image", "text/plain")})
    assert r.status_code == 400
