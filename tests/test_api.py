import pytest
from fastapi.testclient import TestClient
from adv_studio.web.server import app
from adv_studio.data import get_sample_digits
from adv_studio.data.transforms import tensor_to_base64_png


@pytest.fixture
def client():
    return TestClient(app)


def test_api_models(client):
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert len(data["models"]) >= 2


def test_api_attacks(client):
    res = client.get("/api/attacks")
    assert res.status_code == 200
    data = res.json()
    assert "attacks" in data
    assert len(data["attacks"]) >= 5


def test_api_defenses(client):
    res = client.get("/api/defenses")
    assert res.status_code == 200
    data = res.json()
    assert "defenses" in data
    assert len(data["defenses"]) >= 3


def test_api_samples(client):
    res = client.get("/api/samples")
    assert res.status_code == 200
    data = res.json()
    assert "samples" in data
    assert len(data["samples"]) == 10


def test_api_predict_and_attack(client):
    sample_imgs, sample_lbls = get_sample_digits(count=1)
    b64 = tensor_to_base64_png(sample_imgs[0])

    # 1. Test Predict
    pred_res = client.post("/api/predict", json={
        "image_base64": b64,
        "model_name": "clean_cnn"
    })
    assert pred_res.status_code == 200
    pred_data = pred_res.json()
    assert "prediction" in pred_data
    assert "confidence" in pred_data
    assert len(pred_data["probabilities"]) == 10

    # 2. Test Attack (FGSM)
    atk_res = client.post("/api/attack", json={
        "image_base64": b64,
        "true_label": int(sample_lbls[0].item()),
        "model_name": "clean_cnn",
        "attack_name": "fgsm",
        "epsilon": 0.25,
        "steps": 1,
        "targeted": False
    })
    assert atk_res.status_code == 200
    atk_data = atk_res.json()
    assert "adversarial_prediction" in atk_data
    assert "adversarial_image_base64" in atk_data
    assert "perturbation_image_base64" in atk_data
    assert "metrics" in atk_data
    assert "l_inf" in atk_data["metrics"]

    # 3. Test Defense
    adv_b64 = atk_data["adversarial_image_base64"]
    def_res = client.post("/api/defense", json={
        "image_base64": adv_b64,
        "defense_type": "bit_depth",
        "model_name": "clean_cnn",
        "param_value": 3.0
    })
    assert def_res.status_code == 200
    def_data = def_res.json()
    assert "sanitized_image_base64" in def_data
    assert "prediction" in def_data
