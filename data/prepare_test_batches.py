import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def prepare_unlabeled_batch(input_file: str, output_file: str):
    """Removes the 'churn' column to simulate real-time monitoring."""
    df = pd.read_csv(input_file)
    if 'churn' in df.columns:
        df.drop(columns=['churn']).to_csv(output_file, index=False)
        logger.info(f"Created unlabeled batch: {output_file}")
    else:
        logger.warning(f"No 'churn' column in {input_file}")

def create_concept_drift(input_file: str, output_file_labeled: str, output_file_unlabeled: str):
    """
    Simulates Concept Drift: The relationship between Age and Churn changes.
    Suddenly, everyone under 30 churns.
    """
    df = pd.read_csv(input_file)
    
    # Apply the concept drift: Age < 30 -> Churn = 1
    mask = df['age'] < 30
    df.loc[mask, 'churn'] = 1
    
    logger.info(f"Concept Drift: Flipped {mask.sum()} young customers to churn=1.")
    
    # Save labeled version (for retraining)
    df.to_csv(output_file_labeled, index=False)
    logger.info(f"Created labeled concept drift batch: {output_file_labeled}")
    
    # Save unlabeled version (for monitoring)
    df.drop(columns=['churn']).to_csv(output_file_unlabeled, index=False)
    logger.info(f"Created unlabeled concept drift batch: {output_file_unlabeled}")

if __name__ == "__main__":
    # 1. Prepare Data Drift (Inflation) unlabeled file
    prepare_unlabeled_batch(
        "data/batch_2_drifted.csv", 
        "data/batch_2_drifted_unlabeled.csv"
    )
    
    # 2. Create Concept Drift (Young = Churn) files using batch_4
    create_concept_drift(
        "data/batch_4.csv",
        "data/batch_4_concept_drift.csv",
        "data/batch_4_concept_drift_unlabeled.csv"
    )
