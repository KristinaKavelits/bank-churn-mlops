import sys
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

# Add parent directory to path to import logger
sys.path.append(str(Path(__file__).parent.parent))
from src.logger import get_logger

logger = get_logger(__name__)

def train_model(data_path: str, output_predictions_path: str, retrain_path: str = None) -> None:
    """
    Trains a RandomForest model. Handles both Baseline Training and Retraining (Shadow Deployment).
    """
    is_retrain = retrain_path is not None
    
    if is_retrain:
        logger.info(f"Retraining mode triggered with new data: {retrain_path}")
        df = pd.read_csv(retrain_path)
        
        # 1. Graceful Exit for Delayed Ground Truth
        if 'churn' not in df.columns:
            logger.warning(f"⚠️ Retrain batch {retrain_path} lacks 'churn' column. Delayed ground truth. Exiting gracefully.")
            sys.exit(0)
            
        # 2. Train on New Data Only (Avoid Poisoned Data)
        logger.info(f"Discarding historical data to avoid 'Poisoned Data'. Training exclusively on {len(df)} new rows.")
        run_name = "retrain_model"
    else:
        logger.info(f"Baseline training mode using {data_path}")
        df = pd.read_csv(data_path)
        run_name = "baseline_model"

    # Separate features and target
    X = df.drop(columns=['churn'])
    y = df['churn']

    # Split into train and test sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logger.info(f"Data split into train ({len(X_train)} rows) and test ({len(X_test)} rows)")

    # Define preprocessing
    numeric_features = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']
    categorical_features = ['country', 'gender']
    passthrough_features = ['credit_card', 'active_member']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('pass', 'passthrough', passthrough_features)
        ])

    rf_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'class_weight': 'balanced'
    }

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(**rf_params))
    ])

    # Set up MLflow tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("bank_churn_prediction")

    logger.info(f"Starting MLflow run: '{run_name}'")
    with mlflow.start_run(run_name=run_name) as run:
        logger.info("Training the model pipeline...")
        
        # Track execution time
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        execution_time = time.time() - start_time
        logger.info(f"Training completed in {execution_time:.2f} seconds")

        # Evaluate on the test set
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        logger.info(f"Test Set Evaluation - Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        mlflow.log_params(rf_params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        metric_name = "retraining_execution_time" if is_retrain else "baseline_execution_time"
        mlflow.log_metric(metric_name, execution_time)
        
        # 3. Shadow Deployment Metrics & Test Set
        if is_retrain:
            # The f1 calculated above is exactly the challenger_f1 on the 200 unseen new rows
            mlflow.log_metric("challenger_f1", f1)
            logger.info(f"🏆 Challenger F1 (on {len(X_test)}-row holdout test set): {f1:.4f}")
            
            # Save the 200-row holdout test set for registry.py
            test_set_path = Path(output_predictions_path).parent / "staging_test_set.csv"
            staging_test_df = X_test.copy()
            staging_test_df['churn'] = y_test
            staging_test_df.to_csv(test_set_path, index=False)
            logger.info(f"Saved the {len(X_test)}-row holdout test set to {test_set_path} for registry.py evaluation.")

        # Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Retained (0)", "Churned (1)"])
        disp.plot(ax=ax_cm, cmap="Blues", values_format="d")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        mlflow.log_figure(fig_cm, "plots/confusion_matrix.png")
        plt.close(fig_cm)

        # Feature Importance Plot
        cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
        cat_features_out = cat_encoder.get_feature_names_out(categorical_features)
        all_features = numeric_features + list(cat_features_out) + passthrough_features
        importances = pipeline.named_steps['classifier'].feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_features = [all_features[i] for i in indices]
        sorted_importances = importances[indices]

        fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
        ax_fi.bar(range(len(importances)), sorted_importances, align="center", color="skyblue")
        ax_fi.set_xticks(range(len(importances)))
        ax_fi.set_xticklabels(sorted_features, rotation=45, ha="right")
        ax_fi.set_title("Random Forest Feature Importances")
        ax_fi.set_ylabel("Importance Score")
        plt.tight_layout()
        mlflow.log_figure(fig_fi, "plots/feature_importance.png")
        plt.close(fig_fi)

        signature = infer_signature(X_train, pipeline.predict(X_train))

        # Log and Register the model
        model_name = "bank_churn_model"
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            registered_model_name=model_name
        )
        
        # 4. MLflow Model Registry (Production vs Staging using Aliases)
        client = MlflowClient()
        model_version = model_info.registered_model_version
        
        if is_retrain:
            client.set_registered_model_alias(
                name=model_name,
                alias="staging",
                version=model_version
            )
            logger.info(f"Model registered as version {model_version} and assigned alias 'staging' (Shadow Mode)")
            output_path = Path(output_predictions_path).parent / "staging_reference_with_predictions.csv"
            
            # CRITICAL FIX: The Shrinking Baseline
            # To ensure the monitoring baseline stays at a healthy size (1000 rows),
            # we generate predictions on the ENTIRE 1000-row new batch, not just the 200-row test set.
            logger.info(f"Generating predictions on the ENTIRE new batch ({len(df)} rows) for the future monitoring baseline...")
            df_with_preds = df.copy()
            df_with_preds['y_pred'] = pipeline.predict(df.drop(columns=['churn']))
            df_with_preds['y_pred_proba'] = pipeline.predict_proba(df.drop(columns=['churn']))[:, 1]
            
        else:
            client.set_registered_model_alias(
                name=model_name,
                alias="production",
                version=model_version
            )
            logger.info(f"Model registered as version {model_version} and assigned alias 'production'")
            output_path = Path(output_predictions_path)
            
            logger.info(f"Generating predictions on the test dataset ({len(X_test)} rows) for the initial monitoring baseline...")
            df_with_preds = X_test.copy()
            df_with_preds['churn'] = y_test
            df_with_preds['y_pred'] = pipeline.predict(X_test)
            df_with_preds['y_pred_proba'] = pipeline.predict_proba(X_test)[:, 1] 

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_with_preds.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")
    logger.info("✅ Training job completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Retrain the Bank Churn Model")
    parser.add_argument("--data", type=str, default="data/reference.csv", help="Path to historical reference data")
    parser.add_argument("--output", type=str, default="data/reference_with_predictions.csv", help="Path to save predictions")
    parser.add_argument("--retrain", type=str, default=None, help="Path to new batch data for retraining (triggers Shadow Deployment)")
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data, 
        output_predictions_path=args.output,
        retrain_path=args.retrain
    )