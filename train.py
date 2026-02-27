"""
Script to run a training job for a given algorithm and model configuration.

Takes as input a raw CSV file to build (train and test) the model and generates a
new model instance in the model registry folder (archiving the previous one
based on its model_version_id).

Compulsory arguments:
    - path_data: Path to the CSV file containing labeled training data.
    - model_version_id: Version identifier for the trained model (e.g., YYYYMM).

Optional arguments:
    - min_df: Minimum document frequency for text vectorization (default: 3).
    - max_df: Maximum document frequency threshold (default: 0.5).
    - max_features: Max number of features for vectorization (default: None).
"""

import argparse
import os
import pandas as pd
from datetime import datetime
import logging
import sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src import models, config, utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main(
    path_data: str,
    model_version_id: str,
    # TODO: Add your arguments
    min_df: int,
) -> None:
    """
    Main training pipeline.

    Args:
        path_data (str): Path to the raw CSV dataset.
        model_version_id (str): Unique ID for model versioning.
        min_df (int): Minimum document frequency.
        max_df (float): Maximum document frequency.
        max_features (int): Maximum number of features for vectorizer.

    Returns:
        None
    """
    path_interim = os.path.join("data", "interim")
    path_data_train = os.path.join(path_interim, "train.csv")
    path_data_test = os.path.join(path_interim, "test.csv")

    if not os.path.exists(path_interim):
        logging.error("Folder /data/interim/ not found. Please create it before running.")
        return

    # Load  data
    logging.info(f"Reading raw data from: {path_data}")
    df_raw = pd.read_csv(path_data)

    logging.info("Class distribution in FINAL_LABEL:\n%s", df_raw['FINAL_LABEL'].value_counts(normalize=True))

    # Prepar  data
    df_prep = df_raw.copy()
    df_prep['y_is_nf'] =...
    df_prep['x_text'] = ...
    df_prep =...  # keep only ["x_text", "y_is_nf"]

    df_train, df_test = train_test_split(
        ...  # use parameters from config.py !!!!
    )

    logging.info(f"Training data shape: {df_train.shape}")
    logging.info(f"Test data shape: {df_test.shape}")

    df_train.to_csv(path_data_train, index=False)
    df_test.to_csv(path_data_test, index=False)
    logging.info(f"Saved interim train and test data to {path_interim}")

    # Model training
    skl_pl = models.get_model(
       ...  # your selected hiperparamters for the champion architecture
    )

    X_train, y_train = df_train['x_text'], df_train['y_is_nf']
    X_test, y_test = df_test['x_text'], df_test['y_is_nf']

    logging.info("Training model...")
    skl_pl.fit(...)

    # get label predictions
    y_pred_train = ...
    y_pred_test = ...

    # get eval f1-scores
    f1_train = ...
    f1_test = ...

    logging.info(f"F1 Score (Train): {f1_train:.4f}")
    logging.info(f"F1 Score (Test): {f1_test:.4f}")

    # Metadata for model registry
    metadata = {
        "score": config.SCORE,
        "test_value": f1_test,
        "version_id": model_version_id,
        "exe_dt": datetime.now().strftime("%Y%m%d"),
        "sklearn": sklearn.__version__
    }


    utils.register_model(
        skl_pl,
        metadata,
        model_version_id
    )
    logging.info(f"Model registered with version ID: {model_version_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a text classification model.")

    parser.add_argument("path_data", type=str, help="Path to the raw dataset CSV file.")
    parser.add_argument("model_version_id", type=str, help="Model version ID (e.g. 202507).")
    # TODO: Add your arguments
    parser.add_argument("--min_df", type=int, default=3, help="Min document frequency for vectorizer.")


    args = parser.parse_args()

    main(
        args.path_data,
        args.model_version_id,
        # TODO: Add your arguments
        args.min_df,

    )
