import sys
import argparse
from pathlib import Path
import pandas as pd
import shutil
from sklearn.metrics import f1_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Add parent directory to path to import logger
sys.path.append(str(Path(__file__).parent.parent))
from src.logger import get_logger

logger = get_logger(__name__)

def run_registry(test_data_path: str, baseline_path: str) -> None:
    """
    Evaluates the Staging (Challenger) model against the Production (Champion) model.
    Promotes the Challenger if it outperforms the Champion on the holdout test set.
    """
    logger.info(f"Starting Model Registry & Promotion Evaluation...")
    
    # 1. Load Data
    if not Path(test_data_path).exists():
        logger.error(f"🚨 Test data not found at {test_data_path}. Did the retraining job run?")
        sys.exit(1)
        
    logger.info(f"Loading holdout test set from {test_data_path}")
    df_test = pd.read_csv(test_data_path)
    
    if 'churn' not in df_test.columns:
        logger.error("🚨 Test set lacks 'churn' column. Cannot evaluate models.")
        sys.exit(1)
        
    X_test = df_test.drop(columns=['churn'])
    y_test = df_test['churn']

    # Set up MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("bank_churn_prediction")
    client = MlflowClient()
    model_name = "bank_churn_model"

    # 2. Load Models
    logger.info("Loading Champion (Production) and Challenger (Staging) models from MLflow...")
    try:
        champion_model = mlflow.sklearn.load_model(f"models:/{model_name}@production")
        challenger_model = mlflow.sklearn.load_model(f"models:/{model_name}@staging")
    except Exception as e:
        logger.error(f"🚨 Failed to load models from MLflow. Ensure both 'production' and 'staging' aliases exist. Error: {e}")
        sys.exit(1)

    # 3. The Battle (Predict and Calculate F1)
    logger.info("Evaluating Champion model on holdout test set...")
    champion_preds = champion_model.predict(X_test)
    production_f1 = f1_score(y_test, champion_preds)
    
    logger.info("Evaluating Challenger model on holdout test set...")
    challenger_preds = challenger_model.predict(X_test)
    challenger_proba = challenger_model.predict_proba(X_test)[:, 1]
    challenger_f1 = f1_score(y_test, challenger_preds)

    logger.info(f"🥊 THE BATTLE RESULTS 🥊")
    logger.info(f"Champion F1 (Production): {production_f1:.4f}")
    logger.info(f"Challenger F1 (Staging):  {challenger_f1:.4f}")

    # 4. The Promotion Decision
    promote = challenger_f1 > production_f1
    promotion_status = 1 if promote else 0
    f1_improvement = challenger_f1 - production_f1

    with mlflow.start_run(run_name="model_promotion_evaluation"):
        mlflow.log_metric("production_f1", production_f1)
        mlflow.log_metric("challenger_f1", challenger_f1)
        mlflow.log_metric("f1_improvement", f1_improvement)
        mlflow.log_metric("promotion_status", promotion_status)
        
        if promote:
            logger.info("🎉 CHALLENGER WINS! Initiating promotion...")
            
            # Get the version of the staging model
            staging_model_version = client.get_model_version_by_alias(model_name, "staging")
            new_version = staging_model_version.version
            
            # Update the 'production' alias to point to the Challenger's version
            client.set_registered_model_alias(
                name=model_name,
                alias="production",
                version=new_version
            )
            
            # Remove the staging alias since it is now the production model
            client.delete_registered_model_alias(
                name=model_name,
                alias="staging"
            )
            logger.info(f"✅ Model version {new_version} successfully promoted to 'Production' alias (and removed from 'Staging').")
            
            # Overwrite the baseline data for future monitoring
            logger.info("Updating the monitoring baseline with Challenger's predictions...")
            staging_baseline_path = Path(baseline_path).parent / "staging_reference_with_predictions.csv"
            
            if staging_baseline_path.exists():
                shutil.move(str(staging_baseline_path), baseline_path)
                logger.info(f"✅ Overwrote {baseline_path} with new baseline data.")
            else:
                logger.error(f"🚨 Could not find staging baseline at {staging_baseline_path} to promote!")
                sys.exit(1)
            
        else:
            logger.info("🛡️ CHAMPION DEFENDS ITS TITLE! No promotion.")
            logger.info("The production model remains unchanged.")
            
    logger.info("Registry evaluation completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and Promote Models (Shadow Deployment)")
    parser.add_argument("--test-data", type=str, default="data/staging_test_set.csv", help="Path to the holdout test set")
    parser.add_argument("--baseline", type=str, default="data/reference_with_predictions.csv", help="Path to overwrite the monitoring baseline")
    
    args = parser.parse_args()
    
    run_registry(args.test_data, args.baseline)