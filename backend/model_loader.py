"""
backend/model_loader.py
-----------------------
Manages downloading, caching, and loading model artifacts from Hugging Face:
repository 'alfiinyang/XGSupply'.
"""

import os
import urllib.request
import logging

logger = logging.getLogger(__name__)

HF_REPO_ID = "alfiinyang/XGSupply"
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

REQUIRED_FILES = [
    "xgboost_supply_model.joblib",
    "scaler.joblib",
    "training_feature_names.joblib",
    "scaled_column_names.joblib"
]


def ensure_model_files(model_dir: str = DEFAULT_MODEL_DIR) -> dict:
    """
    Ensures all required model artifacts are downloaded and available locally.
    Returns a dict mapping filename to local file path.
    """
    os.makedirs(model_dir, exist_ok=True)
    paths = {}

    for fname in REQUIRED_FILES:
        local_path = os.path.join(model_dir, fname)
        if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
            url = f"https://huggingface.co/{HF_REPO_ID}/resolve/main/{fname}"
            logger.info(f"Downloading {fname} from Hugging Face ({url})...")
            try:
                urllib.request.urlretrieve(url, local_path)
                logger.info(f"Successfully downloaded {fname} ({os.path.getsize(local_path)} bytes)")
            except Exception as e:
                logger.error(f"Failed to download {fname}: {e}")
                raise RuntimeError(f"Could not retrieve {fname} from Hugging Face: {e}")
        paths[fname] = local_path

    return paths


def load_artifacts(model_dir: str = DEFAULT_MODEL_DIR):
    """
    Loads and returns: dict with (model, scaler, train_cols, scaled_cols).
    Includes resilient fallback if ML packages encounter version incompatibilities.
    """
    import joblib

    paths = ensure_model_files(model_dir)

    train_cols = joblib.load(paths["training_feature_names.joblib"])
    scaled_cols = joblib.load(paths["scaled_column_names.joblib"])

    scaler = None
    try:
        scaler = joblib.load(paths["scaler.joblib"])
    except Exception as e:
        logger.warning(f"Note on scaler loading: {e}")

    model = None
    try:
        model = joblib.load(paths["xgboost_supply_model.joblib"])
    except Exception as e:
        logger.warning(f"Note on XGBoost model loading: {e}")

    return {
        "model": model,
        "scaler": scaler,
        "train_cols": train_cols,
        "scaled_cols": scaled_cols
    }
