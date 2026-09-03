import json

import joblib

from app import app


SAMPLE_FORM = {
    "Marital status": "1",
    "Application mode": "1",
    "Application order": "1",
    "Course": "9119",
    "Daytime/evening attendance": "1",
    "Previous qualification": "1",
    "Previous qualification (grade)": "130",
    "Admission grade": "125",
    "Age at enrollment": "19",
    "Mother's qualification": "1",
    "Father's qualification": "1",
    "Displaced": "0",
    "International": "0",
    "Debtor": "0",
    "Tuition fees up to date": "1",
    "Scholarship holder": "0",
    "Gender": "0",
    "Educational special needs": "0",
    "Unemployment rate": "11.6",
    "Inflation rate": "1.2",
    "GDP": "0.0",
}


def test_model_artifact_contract():
    bundle = joblib.load(app.config["MODEL_PATH"])

    assert bundle["positive_label"] == "Dropout"
    assert bundle["negative_label"] == "Not at risk"
    assert bundle["positive_class"] == 1
    assert set(bundle["labels"].values()) == {"Dropout", "Not at risk"}
    assert hasattr(bundle["model"], "predict_proba")
    assert "label_encoder" not in bundle


def test_metrics_include_cv_class_report_and_calibration():
    with open(app.config["METRICS_PATH"]) as f:
        metrics = json.load(f)

    assert metrics["task"] == "binary_dropout_risk"
    assert "cv_results" in metrics
    assert "classification_report" in metrics
    assert "confusion_matrix" in metrics
    assert "brier_score" in metrics["holdout_metrics"]
    assert "Dropout" in metrics["classification_report"]
    assert "Not at risk" in metrics["classification_report"]


def test_home_and_model_info_pages_render():
    client = app.test_client()

    home = client.get("/")
    model_info = client.get("/model-info")

    assert home.status_code == 200
    assert b"Assess a student's dropout risk" in home.data
    assert model_info.status_code == 200
    assert b"Binary dropout-risk model" in model_info.data


def test_about_page_renders():
    client = app.test_client()

    about = client.get("/about")

    assert about.status_code == 200
    assert b"Predicting risk before the first class" in about.data
    assert b"Tech stack" in about.data
    assert b"Models compared" in about.data


def test_predict_route_renders_binary_risk_result():
    client = app.test_client()

    response = client.post("/predict", data=SAMPLE_FORM)

    assert response.status_code == 200
    assert b"Dropout risk:" in response.data
    assert b"Calibrated dropout probability" in response.data
    assert b"binary calibrated output" in response.data
