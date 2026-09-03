# Student Dropout Risk Predictor

A minor project that predicts whether a student is at **dropout risk** using
only information known at the moment of enrollment: application details, prior
qualifications, family background, financial status, and macroeconomic context.

The project uses the UCI "Predict Students' Dropout and Academic Success"
dataset: 4,424 real students with recorded outcomes. The model is now binary:

- `Dropout`
- `Not at risk` (`Enrolled` or `Graduate` in the original dataset)

This binary framing matches the real intervention question better than a
three-class outcome predictor: should an advisor check in with this student?

## Why this scope matters

The raw dataset includes "Curricular units 1st/2nd sem" columns: grades, pass
counts, enrolled units, and evaluations recorded after the student has already
started studying. This project deliberately excludes those columns because they
can leak the outcome. A student who has already dropped out naturally has very
low second-semester activity, so using those columns would inflate performance
while making the tool less useful as an early-warning system.

This version only uses enrollment-time features, which keeps the result honest:
the prediction is made before the student attends a class.

## Current model

`train_model.py` compares three model families with 5-fold stratified
cross-validation:

- Logistic Regression
- Random Forest
- XGBoost

The best model is selected by macro-F1, then wrapped with probability
calibration using `CalibratedClassifierCV`. The Flask app uses the calibrated
model for probabilities and a SHAP explainer from the fitted base model for
local feature explanations.

Current regenerated artifacts selected:

```text
Deployed model: calibrated_random_forest
Holdout accuracy: 79.2%
Holdout macro-F1: 74.5%
Dropout precision: 72.7%
Dropout recall: 56.3%
ROC AUC: 83.6%
Brier score: 0.1449
```

The baseline is about 67.9% holdout accuracy by always predicting `Not at risk`,
so accuracy alone is not enough. Dropout recall, macro-F1, ROC AUC, average
precision, and calibration quality are more informative.

## Project structure

```text
student-risk/
|-- data/
|   `-- dropout_data.csv
|-- model/
|   |-- model.pkl
|   |-- shap_values.pkl
|   |-- feature_names.json
|   `-- metrics.json
|-- static/css/style.css
|-- templates/
|   |-- base.html
|   |-- index.html
|   |-- result.html
|   `-- model_info.html
|-- tests/test_app.py
|-- app.py
|-- config.py
|-- train_model.py
`-- requirements.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Then open `http://127.0.0.1:5000`.

## Configuration

The app no longer hardcodes production-unsafe debug mode. Defaults are safe for
normal local use:

```text
FLASK_DEBUG=false
FLASK_HOST=127.0.0.1
FLASK_PORT=5000
MODEL_PATH=model/model.pkl
SHAP_PATH=model/shap_values.pkl
METRICS_PATH=model/metrics.json
```

For local debug mode:

```bash
set FLASK_DEBUG=true
python app.py
```

## Tests

```bash
python -m pytest -q
```

The tests cover:

- binary model artifact contract
- metrics file structure
- home and model-info pages
- sample prediction route

## Limitations

This is still a screening aid, not a final decision tool.

- Enrollment-time data alone cannot capture everything that causes dropout.
- Dropout recall is useful but still misses some true dropout cases.
- Sensitive or proxy-sensitive fields such as debtor status, scholarship,
  gender, special needs, parental qualification, and international status should
  be handled carefully.
- A production version should add fairness checks and early-semester signals
  such as attendance, first assessments, LMS activity, and advisor notes.

Use the result as a prompt for supportive outreach, not as a verdict.
