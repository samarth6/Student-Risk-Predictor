import os


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    DEBUG = _env_bool("FLASK_DEBUG", default=False)
    HOST = os.getenv("FLASK_HOST", "127.0.0.1")
    PORT = int(os.getenv("FLASK_PORT", "5000"))
    MODEL_PATH = os.getenv("MODEL_PATH", "model/model.pkl")
    SHAP_PATH = os.getenv("SHAP_PATH", "model/shap_values.pkl")
    METRICS_PATH = os.getenv("METRICS_PATH", "model/metrics.json")
