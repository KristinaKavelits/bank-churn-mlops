import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path to import logger
sys.path.append(str(Path(__file__).parent.parent))
from src.logger import get_logger

logger = get_logger(__name__)

def ingest_data(input_path: str, output_dir: str) -> None:
    """
    Loads raw data, cleans it, and splits it into reference and batch files.
    """
    logger.info(f"Starting data ingestion from {input_path}")
    
    # 1. Load data
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Successfully loaded {len(df)} rows.")
    except FileNotFoundError:
        logger.error(f"🚨 File not found: {input_path}")
        sys.exit(1)

    # 2. Clean data
    columns_to_drop = ['customer_id', 'surname']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    logger.info(f"Dropped columns: {columns_to_drop}")

    # 3. Create output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 4. Split logic (6000 reference, 4 batches of 1000)
    # This gives us a massive 6000-row baseline (4800 train / 1200 test)
    # The 1200-row test set becomes our stable monitoring baseline!
    reference_df = df.iloc[:6000]
    reference_path = out_path / "reference.csv"
    reference_df.to_csv(reference_path, index=False)
    logger.info(f"Saved reference data ({len(reference_df)} rows) to {reference_path}")

    remaining_df = df.iloc[6000:]
    batch_size = 1000
    
    # We only need 4 batches now (1000 rows each)
    for i in range(4):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_df = remaining_df.iloc[start_idx:end_idx]
        
        batch_path = out_path / f"batch_{i+1}.csv"
        batch_df.to_csv(batch_path, index=False)
        logger.info(f"Saved batch {i+1} ({len(batch_df)} rows) to {batch_path}")

    logger.info("✅ Data ingestion completed successfully.")

if __name__ == "__main__":
    ingest_data("data/bank_customer_churn.csv", "data/")
