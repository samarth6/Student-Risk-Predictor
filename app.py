"""
app.py
------
Flask backend for the Student Dropout Risk Predictor (real UCI dropout
dataset, 4,424 students, enrollment-time features only). The deployed task is
binary: Dropout vs Not at risk.

Routes:
    GET  /            -> input form
    POST /predict      -> runs the model, returns results page with SHAP explanation
    GET  /model-info    -> model comparison table
"""

import json

import joblib
import numpy as np
import pandas as pd
from flask import Flask, abort, render_template, request

from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# ---------------------------------------------------------------------------
# Load model artifacts once at startup
# ---------------------------------------------------------------------------
bundle = joblib.load(app.config["MODEL_PATH"])
MODEL = bundle["model"]
MODEL_NAME = bundle["model_name"]
PREPROCESSOR = bundle["preprocessor"]
NUMERIC_FEATURES = bundle["numeric_features"]
BINARY_FEATURES = bundle["binary_features"]
CATEGORICAL_FEATURES = bundle["categorical_features"]
QUALIFICATION_BAND = bundle["qualification_band"]
LABELS = {int(k): v for k, v in bundle["labels"].items()}
POSITIVE_CLASS = int(bundle["positive_class"])
POSITIVE_LABEL = bundle["positive_label"]
NEGATIVE_LABEL = bundle["negative_label"]
# Tuned classification threshold (from threshold sweep in train_model.py)
DECISION_THRESHOLD = float(bundle.get("decision_threshold", 0.50))

shap_bundle = joblib.load(app.config["SHAP_PATH"])
EXPLAINER = shap_bundle["explainer"]

with open(app.config["METRICS_PATH"]) as f:
    METRICS = json.load(f)

RISK_COLORS = {NEGATIVE_LABEL: "#2f9e44", POSITIVE_LABEL: "#d64545"}

# Human-readable labels for the SHAP chart
FRIENDLY_NAMES = {
    "num__Application order": "Application order (choice rank)",
    "num__Previous qualification (grade)": "Previous qualification grade",
    "num__Admission grade": "Admission grade",
    "num__Age at enrollment": "Age at enrollment",
    "num__Unemployment rate": "Unemployment rate (enrollment year)",
    "num__Inflation rate": "Inflation rate (enrollment year)",
    "num__GDP": "GDP (enrollment year)",
    "bin__Daytime/evening attendance": "Attendance: daytime",
    "bin__Displaced": "Displaced from home",
    "bin__Educational special needs": "Educational special needs",
    "bin__Debtor": "Debtor",
    "bin__Tuition fees up to date": "Tuition fees up to date",
    "bin__Gender": "Gender: male",
    "bin__Scholarship holder": "Scholarship holder",
    "bin__International": "International student",
}


def friendly_feature_name(raw_name: str) -> str:
    if raw_name in FRIENDLY_NAMES:
        return FRIENDLY_NAMES[raw_name]
    if raw_name.startswith("cat__"):
        stripped = raw_name[len("cat__"):]
        for feat in CATEGORICAL_FEATURES:
            prefix = feat + "_"
            if stripped.startswith(prefix):
                value = stripped[len(prefix):]
                short_feat = feat.replace(" band", "")
                return f"{short_feat}: {value}"
    return raw_name


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/model-info")
def model_info():
    return render_template("model_info.html", metrics=METRICS, best_model=MODEL_NAME)


@app.route("/about")
def about():
    return render_template("about.html", metrics=METRICS)


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    try:
        row = {
            "Application order": int(form["Application order"]),
            "Previous qualification (grade)": float(form["Previous qualification (grade)"]),
            "Admission grade": float(form["Admission grade"]),
            "Age at enrollment": int(form["Age at enrollment"]),
            "Unemployment rate": float(form["Unemployment rate"]),
            "Inflation rate": float(form["Inflation rate"]),
            "GDP": float(form["GDP"]),
            "Daytime/evening attendance": int(form["Daytime/evening attendance"]),
            "Displaced": int(form["Displaced"]),
            "Educational special needs": int(form["Educational special needs"]),
            "Debtor": int(form["Debtor"]),
            "Tuition fees up to date": int(form["Tuition fees up to date"]),
            "Gender": int(form["Gender"]),
            "Scholarship holder": int(form["Scholarship holder"]),
            "International": int(form["International"]),
            "Marital status": form["Marital status"],
            "Application mode": form["Application mode"],
            "Course": form["Course"],
            "Previous qualification band": QUALIFICATION_BAND.get(
                int(form["Previous qualification"]), "Other/unknown"
            ),
            "Mother's qualification band": QUALIFICATION_BAND.get(
                int(form["Mother's qualification"]), "Other/unknown"
            ),
            "Father's qualification band": QUALIFICATION_BAND.get(
                int(form["Father's qualification"]), "Other/unknown"
            ),
        }
    except (KeyError, ValueError) as exc:
        abort(400, description=f"Invalid or missing form field: {exc}")

    X = pd.DataFrame([row])[NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES]
    X_t = PREPROCESSOR.transform(X)

    proba = MODEL.predict_proba(X_t)[0]
    class_to_index = {int(cls): idx for idx, cls in enumerate(MODEL.classes_)}
    dropout_probability = round(float(proba[class_to_index[POSITIVE_CLASS]]) * 100, 1)
    safe_probability = round(100 - dropout_probability, 1)
    # Use the tuned threshold instead of the default 0.5
    pred_encoded = POSITIVE_CLASS if (dropout_probability / 100) >= DECISION_THRESHOLD else (1 - POSITIVE_CLASS)
    pred_label = LABELS[pred_encoded]
    proba_map = {
        POSITIVE_LABEL: dropout_probability,
        NEGATIVE_LABEL: safe_probability,
    }

    shap_out = EXPLAINER.shap_values(X_t)
    feature_names = PREPROCESSOR.get_feature_names_out()

    if isinstance(shap_out, list):
        class_shap = np.array(shap_out[POSITIVE_CLASS][0])
    else:
        arr = np.array(shap_out)
        if arr.ndim == 3:
            class_shap = arr[0, :, POSITIVE_CLASS]
        else:
            class_shap = arr[0]

    contributions = list(zip(feature_names, class_shap))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_contributions = contributions[:6]

    chart_labels = [friendly_feature_name(name) for name, _ in top_contributions]
    chart_values = [round(float(val), 3) for _, val in top_contributions]
    chart_colors = ["#d64545" if v > 0 else "#2f9e44" for v in chart_values]

    return render_template(
        "result.html",
        risk=pred_label,
        risk_color=RISK_COLORS[pred_label],
        proba_map=proba_map,
        positive_label=POSITIVE_LABEL,
        negative_label=NEGATIVE_LABEL,
        dropout_probability=dropout_probability,
        chart_labels=json.dumps(chart_labels),
        chart_values=json.dumps(chart_values),
        chart_colors=json.dumps(chart_colors),
        model_name=MODEL_NAME,
        input_summary=row,
    )


if __name__ == "__main__":
    app.run(
        debug=app.config["DEBUG"],
        host=app.config["HOST"],
        port=app.config["PORT"],
    )
