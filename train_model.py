
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, train_test_split, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
POSITIVE_LABEL = "Dropout"
NEGATIVE_LABEL = "Not at risk"
LABELS = {0: NEGATIVE_LABEL, 1: POSITIVE_LABEL}
MODEL_DIR = Path("model")


df = pd.read_csv("data/dropout_data.csv", sep=";", encoding="utf-8-sig")
df.columns = [c.strip() for c in df.columns]


QUALIFICATION_BAND = {
    1: "Secondary", 2: "Higher education (bachelor/degree)", 3: "Higher education (bachelor/degree)",
    4: "Higher education (master/doctorate)", 5: "Higher education (master/doctorate)",
    6: "Higher education (bachelor/degree)", 9: "Secondary", 10: "Secondary", 11: "Basic or below",
    12: "Secondary", 13: "Secondary", 14: "Secondary", 15: "Secondary", 18: "Secondary",
    19: "Basic or below", 20: "Secondary", 22: "Secondary", 25: "Secondary", 26: "Basic or below",
    27: "Secondary", 29: "Basic or below", 30: "Basic or below", 31: "Secondary", 33: "Secondary",
    34: "Other/unknown", 35: "Basic or below", 36: "Basic or below", 37: "Basic or below",
    38: "Basic or below", 39: "Higher education (bachelor/degree)", 40: "Higher education (bachelor/degree)",
    41: "Higher education (bachelor/degree)", 42: "Higher education (bachelor/degree)",
    43: "Higher education (master/doctorate)", 44: "Higher education (master/doctorate)",
}

for col in ["Previous qualification", "Mother's qualification", "Father's qualification"]:
    df[col + " band"] = df[col].map(QUALIFICATION_BAND).fillna("Other/unknown")


NUMERIC_FEATURES = [
    "Application order", "Previous qualification (grade)", "Admission grade",
    "Age at enrollment", "Unemployment rate", "Inflation rate", "GDP",
]
BINARY_FEATURES = [
    "Daytime/evening attendance", "Displaced", "Educational special needs",
    "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "International",
]
CATEGORICAL_FEATURES = [
    "Marital status", "Application mode", "Course",
    "Previous qualification band", "Mother's qualification band", "Father's qualification band",
]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

X = df[FEATURES].copy()
y = (df["Target"] == POSITIVE_LABEL).astype(int)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ]
)

baseline_models = {
    "logistic_regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "xgboost": XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        eval_metric="logloss",
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=RANDOM_STATE,
    ),
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}

print("=" * 60)
print("STEP 1 - Baseline cross-validation comparison")
print("=" * 60)

cv_results = {}
for name, estimator in baseline_models.items():
    pipe = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", estimator),
        ]
    )
    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    cv_results[name] = {
        metric: {
            "mean": round(float(np.mean(scores[f"test_{metric}"])), 4),
            "std": round(float(np.std(scores[f"test_{metric}"])), 4),
        }
        for metric in scoring
    }
    print(f"  {name:22s} -> macro-F1 {cv_results[name]['f1_macro']}")

best_name = max(cv_results, key=lambda k: cv_results[k]["f1_macro"]["mean"])
print(f"\n  Best baseline model: {best_name} (macro-F1 = {cv_results[best_name]['f1_macro']['mean']})")


print("\n" + "=" * 60)
print(f"STEP 2 - Optuna hyperparameter tuning ({best_name}, 50 trials)")
print("=" * 60)

preprocessor.fit(X_train)
X_train_t = preprocessor.transform(X_train)
X_test_t = preprocessor.transform(X_test)
feature_names = list(preprocessor.get_feature_names_out())

pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

def make_objective(model_name):
    def objective(trial):
        if model_name == "random_forest":
            estimator = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 600),
                max_depth=trial.suggest_int("max_depth", 4, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        elif model_name == "xgboost":
            estimator = XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 600),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                scale_pos_weight=pos_weight,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
            )
        else:  # logistic_regression
            estimator = LogisticRegression(
                C=trial.suggest_float("C", 0.001, 10.0, log=True),
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )

        pipe = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", estimator),
        ])
        scores = cross_val_score(
            pipe, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring="f1_macro",
            n_jobs=1,
            error_score="raise",
        )
        mean_score = float(np.mean(scores))
        if not np.isfinite(mean_score) or not np.all(np.isfinite(scores)):
            trial.set_user_attr("cv_scores", scores.tolist())
            raise optuna.TrialPruned("Cross-validation returned a non-finite score.")

        return mean_score

    return objective

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(make_objective(best_name), n_trials=50, show_progress_bar=False)

best_params = study.best_params
best_trial_f1 = round(study.best_value, 4)
print(f"  Best Optuna macro-F1 (CV): {best_trial_f1}")
print(f"  Best params: {best_params}")


if best_name == "random_forest":
    tuned_base = RandomForestClassifier(
        **best_params, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
elif best_name == "xgboost":
    tuned_base = XGBClassifier(
        **best_params, scale_pos_weight=pos_weight,
        eval_metric="logloss", random_state=RANDOM_STATE,
    )
else:
    tuned_base = LogisticRegression(
        **best_params, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE,
    )


print("\n" + "=" * 60)
print("STEP 3 - Fitting calibrated model on train set")
print("=" * 60)

base_model = clone(tuned_base)
base_model.fit(X_train_t, y_train)

calibrated_model = CalibratedClassifierCV(
    estimator=clone(tuned_base),
    method="sigmoid",
    cv=5,
)
calibrated_model.fit(X_train_t, y_train)


print("\n" + "=" * 60)
print("STEP 4 - Threshold sweep (0.30 -> 0.65)")
print("=" * 60)

cv_calibrated = CalibratedClassifierCV(
    estimator=clone(tuned_base),
    method="sigmoid",
    cv=3,
)
proba_train_oof = cross_val_predict(
    cv_calibrated, X_train_t, y_train, cv=5, method="predict_proba", n_jobs=1
)[:, 1]

thresholds = np.arange(0.30, 0.66, 0.01)
best_threshold = 0.50
best_f1 = 0.0

print(f"  {'Threshold':>10}  {'Macro-F1':>10}  {'Dropout Recall':>15}  {'Dropout Precision':>18}")
for t in thresholds:
    preds_t = (proba_train_oof >= t).astype(int)
    f1 = f1_score(y_train, preds_t, average="macro", zero_division=0)
    rec = recall_score(y_train, preds_t, pos_label=1, zero_division=0)
    prec = precision_score(y_train, preds_t, pos_label=1, zero_division=0)
    marker = " <-- best" if f1 > best_f1 else ""
    print(f"  {t:>10.2f}  {f1:>10.4f}  {rec:>15.4f}  {prec:>18.4f}{marker}")
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = round(float(t), 2)

print(f"\n  Chosen threshold: {best_threshold}  (macro-F1 = {best_f1:.4f})")


proba_test = calibrated_model.predict_proba(X_test_t)[:, 1]
preds = (proba_test >= best_threshold).astype(int)

holdout_metrics = {
    "accuracy": round(accuracy_score(y_test, preds), 4),
    "precision_macro": round(precision_score(y_test, preds, average="macro", zero_division=0), 4),
    "recall_macro": round(recall_score(y_test, preds, average="macro", zero_division=0), 4),
    "f1_macro": round(f1_score(y_test, preds, average="macro", zero_division=0), 4),
    "dropout_precision": round(precision_score(y_test, preds, pos_label=1, zero_division=0), 4),
    "dropout_recall": round(recall_score(y_test, preds, pos_label=1, zero_division=0), 4),
    "dropout_f1": round(f1_score(y_test, preds, pos_label=1, zero_division=0), 4),
    "roc_auc": round(roc_auc_score(y_test, proba_test), 4),
    "average_precision": round(average_precision_score(y_test, proba_test), 4),
    "brier_score": round(brier_score_loss(y_test, proba_test), 4),
    "decision_threshold": best_threshold,
}

print("\nHoldout metrics (with tuned threshold):")
print(json.dumps(holdout_metrics, indent=2))

report = classification_report(
    y_test,
    preds,
    target_names=[NEGATIVE_LABEL, POSITIVE_LABEL],
    output_dict=True,
    zero_division=0,
)
report = {
    label: {
        metric: round(float(value), 4)
        for metric, value in values.items()
        if metric != "support"
    } | {"support": int(values["support"])}
    for label, values in report.items()
    if isinstance(values, dict)
}
matrix = confusion_matrix(y_test, preds, labels=[0, 1]).tolist()
baseline_accuracy = round(float((y_test == 0).mean()), 4)

print("\nClass-level report:")
print(json.dumps(report, indent=2))

# ---------------------------------------------------------------------------
# 10. SHAP explainer for the selected base model
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5 - Building SHAP explainer")
print("=" * 60)

background = shap.sample(X_train_t, 100, random_state=RANDOM_STATE)

if best_name == "logistic_regression":
    explainer = shap.LinearExplainer(base_model, background)
else:
    explainer = shap.TreeExplainer(base_model)

print("  Done.")


print("\n" + "=" * 60)
print("STEP 6 - Saving model artifacts")
print("=" * 60)

MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(
    {
        "model": calibrated_model,
        "explanation_model": base_model,
        "model_name": f"calibrated_{best_name}",
        "base_model_name": best_name,
        "preprocessor": preprocessor,
        "numeric_features": NUMERIC_FEATURES,
        "binary_features": BINARY_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "qualification_band": QUALIFICATION_BAND,
        "labels": LABELS,
        "positive_class": 1,
        "positive_label": POSITIVE_LABEL,
        "negative_label": NEGATIVE_LABEL,
        "decision_threshold": best_threshold,
        "optuna_best_params": best_params,
        "optuna_best_cv_f1": best_trial_f1,
    },
    MODEL_DIR / "model.pkl",
)

joblib.dump({"explainer": explainer, "background": background}, MODEL_DIR / "shap_values.pkl")

with open(MODEL_DIR / "feature_names.json", "w") as f:
    json.dump(feature_names, f, indent=2)

with open(MODEL_DIR / "metrics.json", "w") as f:
    json.dump(
        {
            "task": "binary_dropout_risk",
            "positive_label": POSITIVE_LABEL,
            "negative_label": NEGATIVE_LABEL,
            "best_model": best_name,
            "deployed_model": f"calibrated_{best_name}",
            "decision_threshold": best_threshold,
            "optuna_best_params": best_params,
            "cv_results": cv_results,
            "holdout_metrics": holdout_metrics,
            "classification_report": report,
            "confusion_matrix": {
                "labels": [NEGATIVE_LABEL, POSITIVE_LABEL],
                "matrix": matrix,
            },
            "baseline": {
                "strategy": f"always predict {NEGATIVE_LABEL}",
                "accuracy": baseline_accuracy,
            },
        },
        f,
        indent=2,
    )

print("  Saved model/model.pkl")
print("  Saved model/shap_values.pkl")
print("  Saved model/feature_names.json")
print("  Saved model/metrics.json")
print(f"\n  Decision threshold used: {best_threshold}")
print(f"  Optuna best params:      {best_params}")
