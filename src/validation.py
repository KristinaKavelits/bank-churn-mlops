import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.logger import get_logger

import argparse
import pandas as pd
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfMissingValues,
    TestColumnValueMin,
    TestColumnValueMax
)

logger = get_logger(__name__)

def run_validation(reference_path: str, current_path: str, report_dir: str) -> None:
    """
    Runs technical schema validation using Evidently AI TestSuite.
    Compares the current batch against the reference dataset to ensure structural integrity.

    Args:
        reference_path (str): Path to the reference dataset.
        current_path (str): Path to the current batch dataset to validate.
        report_dir (str): Directory to save the HTML validation report.
    """
    logger.info(f"Loading reference data from {reference_path}")
    reference_data = pd.read_csv(reference_path)
    
    logger.info(f"Loading current batch data from {current_path}")
    current_data = pd.read_csv(current_path)

    # LAYER 1: Pandas Pre-Validation (Fail-Fast before Evidently)
    logger.info("Running Layer 1: Pandas Pre-Validation...")
    
    # Check 1: Missing Columns (Allow 'churn' to be missing for delayed ground truth scenarios)
    missing_cols = set(reference_data.columns) - set(current_data.columns) - {'churn'}
    extra_cols = set(current_data.columns) - set(reference_data.columns)
    
    if missing_cols or extra_cols:
        logger.error("🚨 Layer 1 Validation FAILED: Column mismatch detected!")
        if missing_cols:
            logger.error(f"Missing columns in current batch: {missing_cols}")
        if extra_cols:
            logger.error(f"Unexpected extra columns in current batch: {extra_cols}")
        logger.critical("Exiting pipeline with code 1.")
        sys.exit(1)
        
    # Check 2: Data Type Mismatch
    # If a user types "forty" instead of 40, Pandas will load the column as 'object' instead of 'int64'
    type_mismatches = []
    for col in current_data.columns:
        if col not in reference_data.columns:
            continue
            
        ref_type = reference_data[col].dtype
        curr_type = current_data[col].dtype
        
        # We allow int/float mixing (e.g. 40.0 vs 40) but block object/numeric mixing
        if pd.api.types.is_numeric_dtype(ref_type) and not pd.api.types.is_numeric_dtype(curr_type):
            type_mismatches.append(f"Column '{col}' expected numeric ({ref_type}) but got {curr_type}")
            
    if type_mismatches:
        logger.error("🚨 Layer 1 Validation FAILED: Data type mismatch detected!")
        for mismatch in type_mismatches:
            logger.error(mismatch)
        logger.critical("Exiting pipeline with code 1.")
        sys.exit(1)
        
    logger.info("✅ Layer 1 Passed: Schema and basic types match.")

    # LAYER 2: Evidently TestSuite (Deep Data Quality & HTML Report)
    logger.info("Initializing Layer 2: Evidently TestSuite for Deep Data Quality...")
    
    # If current data doesn't have churn, drop it from reference so Evidently doesn't crash or complain
    if 'churn' not in current_data.columns and 'churn' in reference_data.columns:
        reference_data = reference_data.drop(columns=['churn'])

    # Define the tests we want to run
    data_quality_suite = TestSuite(tests=[
        # 3. Check for missing values (we expect 0 missing values based on our model's needs)
        TestNumberOfMissingValues(),
        
        # 4. Semantic Sanity Check (Business Logic): Age must be between 14 and 120
        TestColumnValueMin(column_name='age', gte=14),
        TestColumnValueMax(column_name='age', lte=120)
    ])

    logger.info("Running validation tests...")
    data_quality_suite.run(reference_data=reference_data, current_data=current_data)

    # Save the visual HTML report with a timestamp to avoid overwriting
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_name = Path(current_path).stem
    report_filename = f"validation_report_{batch_name}_{timestamp}.html"
    report_path = Path(report_dir) / report_filename
    
    report_path.parent.mkdir(parents=True, exist_ok=True)
    data_quality_suite.save_html(str(report_path))
    logger.info(f"Saved validation HTML report to {report_path}")

    # Check if any tests failed
    test_results = data_quality_suite.as_dict()
    failed_tests = test_results['summary']['failed_tests']
    total_tests = test_results['summary']['total_tests']

    if failed_tests > 0:
        logger.error(f"🚨 Validation FAILED! {failed_tests}/{total_tests} tests failed.")
        
        # Log exactly which tests failed
        for test in test_results['tests']:
            if test['status'] == 'FAIL':
                logger.error(f"Failed Test: {test['name']} - {test['description']}")
                
        logger.critical("Exiting pipeline with code 1 due to validation failure.")
        sys.exit(1)
    else:
        logger.info(f"✅ Validation PASSED! All {total_tests} tests successful.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Data Validation")
    parser.add_argument("--ref", type=str, default="data/reference.csv", help="Path to reference data")
    parser.add_argument("--current", type=str, default="data/batch_1.csv", help="Path to current batch data")
    parser.add_argument("--report-dir", type=str, default="reports/", help="Directory to save reports")
    
    args = parser.parse_args()
    
    run_validation(args.ref, args.current, args.report_dir)