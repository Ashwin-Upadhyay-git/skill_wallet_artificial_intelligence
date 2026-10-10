from app import app


def test_predict_rejects_invalid_input():
    client = app.test_client()

    response = client.post(
        "/api/predict",
        json={
            "gender": "0",
            "age": "35",
            "hemoglobin": "-2",
            "mch": "27.4",
            "mchc": "33.1",
            "mcv": "86.0",
        },
    )

    assert response.status_code == 400
    assert "plausible numeric value" in response.get_json()["error"]


def test_predict_accepts_dataset_gender_code():
    client = app.test_client()

    response = client.post(
        "/api/predict",
        json={
            "gender": "0",
            "age": "35",
            "hemoglobin": "12.8",
            "mch": "27.4",
            "mchc": "33.1",
            "mcv": "86.0",
        },
    )

    assert response.status_code == 200
    assert response.get_json()["class"] in {0, 1}
    assert 0 <= response.get_json()["probability"] <= 1
