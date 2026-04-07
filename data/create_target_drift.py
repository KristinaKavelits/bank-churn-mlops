import pandas as pd
from pathlib import Path
import logging
import numpy as np

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def create_target_drift(input_file: str, output_file: str, drift_ratio: float = 0.4):
    """
    Simulates Target (Label) Drift by flipping a percentage of '0' (Retained) labels to '1' (Churned).
    This simulates a massive economic event where people suddenly leave the bank.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        logger.error(f"Input file {input_path} does not exist.")
        return

    logger.info(f"Loading original data from {input_path}")
    df = pd.read_csv(input_path)
    
    if 'churn' not in df.columns:
        logger.error("The 'churn' column is missing from the input data.")
        return

    # Find all the people who originally stayed (churn == 0)
    retained_indices = df[df['churn'] == 0].index
    
    # Calculate how many we need to flip
    num_to_flip = int(len(retained_indices) * drift_ratio)
    
    # Randomly select them and flip them to 1
    np.random.seed(42) # For reproducibility
    indices_to_flip = np.random.choice(retained_indices, size=num_to_flip, replace=False)
    
    df.loc[indices_to_flip, 'churn'] = 1
    
    logger.info(f"Flipped {num_to_flip} labels from 0 to 1 to simulate Target Drift.")
    logger.info(f"New Churn Rate: {(df['churn'].mean() * 100):.1f}%")
    
    # Save the drifted dataset
    df.to_csv(output_path, index=False)
    logger.info(f"Successfully saved target-drifted data to {output_path}")

if __name__ == "__main__":
    # Define paths relative to the project root
    input_csv = "data/batch_3.csv"
    output_csv = "data/batch_3_target_drift.csv"
    
    create_target_drift(input_csv, output_csv, drift_ratio=0.4)
