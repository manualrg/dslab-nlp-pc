"""
Script to run a prediction job using the latest production model.

Reads a CSV file containing a 'REQUIREMENT' column, transforms it into `x_text`,
and generates predictions stored in a new column `y_hats`.

Arguments:
    - path_data: Path to the input CSV file.
    - output_file: Output CSV file name with predictions.

Model is loaded from: models/prod/model.pkl
Output is saved to: data/processed/
"""

import argparse
import os
import pandas as pd
import pickle
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def main(path_data: str, output_file: str) -> None:
    """
    Main function to load a production model and generate predictions.

    Args:
        path_data (str): Path to the input CSV with a 'REQUIREMENT' column.
        output_file (str): File name to save predictions (in /data/processed/).

    Returns:
        None
    """
    
    path_processed = os.path.join("data", "processed")
    path_model_prod = os.path.join("models", "prod")

    if not os.path.exists(path_processed):
        logging.error("Directory /data/processed/ not found. Please create it before running.")
        return

    if not os.path.isfile(path_data):
        logging.error(f"Input file does not exist: {path_data}")
        return

    if not os.path.isfile(os.path.join(path_model_prod, "model.pkl")):
        logging.error(f"Production model not found at: {path_model_prod}")
        return
    
    # Load  data
    logging.info(f"Reading input data from: {path_data}")
    df_score = pd....

    if 'REQUIREMENT' not in df_score.columns:
        logging.error("Missing required column: 'REQUIREMENT' in input file.")
        return

    # Prepar  data
    df_prep = df_score.copy()
    df_prep['x_text'] = ...
    logging.info(f"Input data shape: {df_prep.shape}")

    # Load production model
    logging.info(f"Loading production model from: {path_model_prod}")
    with open(os.path.join(path_model_prod, "model.pkl"), "rb") as file:
        skl_pl = pickle.load( file)
    
    # Get predictions
    logging.info("Generating predictions...")
    X_score = ...
    df_prep['y_hats'] = ...

    # Write predictions
    logging.info(f"Predictions saved to: {path_processed}")
    df_prep.to_csv(
        os.path.join(path_processed, output_file),
        index=False)

    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script load prod model and predict on unseen batch data.")

    parser.add_argument("path_data", type=str,
                        help="Path to the input data, e.g. ./data/raw/my_file.csv")
    parser.add_argument("output_file", type=str,
                        help="Scored file name, e.g. scoring_YYYYMM.csv")

    args = parser.parse_args()
    main(
        args.path_data,
        args.output_file,
    )


