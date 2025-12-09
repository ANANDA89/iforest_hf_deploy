import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------
# 1. Configure logging
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("iforest_training.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

FEATURES = ["radius", "texture", "perimeter", "area"]


def train_and_save_iforest():
    logger.info("Loading breast cancer dataset from sklearn...")
    data = load_breast_cancer()
    X_raw = data.data[:, :4]  # use first 4 features
    logger.info(f"Raw feature matrix shape: {X_raw.shape}")

    # Put into a DataFrame with friendly names
    df = pd.DataFrame(X_raw, columns=FEATURES)
    logger.info("Columns used for training: %s", FEATURES)

    # We treat the problem as unsupervised anomaly detection
    # (IsolationForest uses only X, no labels).
    X_train, X_test = train_test_split(
        df.values, test_size=0.2, random_state=42
    )
    logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Build pipeline: StandardScaler + IsolationForest
    logger.info("Building IsolationForest pipeline...")
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("iforest", IsolationForest(
                n_estimators=200,
                contamination=0.05,  # ~5% anomalies
                random_state=42,
            )),
        ]
    )

    logger.info("Fitting IsolationForest...")
    pipeline.fit(X_train)
    logger.info("Model training completed.")

    # Evaluate roughly on test set
    scores = pipeline.decision_function(X_test)
    logger.info(
        "Sample of decision function scores on test data: "
        f"min={scores.min():.3f}, max={scores.max():.3f}"
    )

    # Save model
    model_path = "iforest_model.pkl"
    logger.info("Saving model to '%s'...", model_path)
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    logger.info("Model saved successfully.")


if __name__ == "__main__":
    logger.info("=== Starting Isolation Forest training script ===")
    train_and_save_iforest()
    logger.info("=== Training script finished successfully ===")
