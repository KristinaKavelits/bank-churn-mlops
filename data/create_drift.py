import pandas as pd
from pathlib import Path
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def create_inflation_drift(input_file: str, output_file: str, multiplier: float = 1.5):
    """
    Simulates an inflation scenario by multiplying financial columns by a given factor.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        logger.error(f"Input file {input_path} does not exist.")
        return

    logger.info(f"Loading original data from {input_path}")
    df = pd.read_csv(input_path)
    
    # Apply drift to financial columns
    logger.info(f"Applying {multiplier}x multiplier to 'balance' and 'estimated_salary'...")
    df['balance'] = df['balance'] * multiplier
    df['estimated_salary'] = df['estimated_salary'] * multiplier
    
    # Save the drifted dataset
    df.to_csv(output_path, index=False)
    logger.info(f"Successfully saved drifted data to {output_path}")

if __name__ == "__main__":
    # Define paths relative to the project root
    input_csv = "data/batch_2.csv"
    output_csv = "data/batch_2_drifted.csv"
    
    create_inflation_drift(input_csv, output_csv, multiplier=1.5)
