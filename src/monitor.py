import sys
import json
import time
from pathlib import Path
from datetime import datetime
import argparse
import pandas as pd
from sklearn.metrics import f1_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import nannyml as nml
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping

# Add parent directory to path to import logger
sys.path.append(str(Path(__file__).parent.parent))
from src.logger import get_logger

logger = get_logger(__name__)

def load_latest_model():
    """Loads the latest trained model from MLflow using the production alias."""
    mlflow.set_tracking_uri("file:./mlruns")
    return mlflow.sklearn.load_model("models:/bank_churn_model@production")

def get_baseline_f1():
    """Retrieves the baseline F1-score from the production model run."""
    client = MlflowClient()
    model_version = client.get_model_version_by_alias("bank_churn_model", "production")
    run = client.get_run(model_version.run_id)
    return run.data.metrics.get("f1_score")

def run_monitoring(reference_path: str, current_path: str, report_dir: str, drift_threshold: float, f1_drop_threshold: float) -> None:
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_name = Path(current_path).stem
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load Data (Simplified: only use reference_with_predictions.csv)
    logger.info(f"Loading reference data from {reference_path}")
    ref_df = pd.read_csv(reference_path)
    
    logger.info(f"Loading current batch data from {current_path}")
    curr_df = pd.read_csv(current_path)

    has_ground_truth = 'churn' in curr_df.columns
    logger.info(f"Ground truth present: {has_ground_truth}")

    # 2. Inference on Current Batch
    logger.info("Running inference on current batch...")
    model = load_latest_model()
    
    # Drop churn for prediction if it exists
    features_df = curr_df.drop(columns=['churn']) if has_ground_truth else curr_df
    curr_df['y_pred'] = model.predict(features_df)
    curr_df['y_pred_proba'] = model.predict_proba(features_df)[:, 1]

    # 3. Evidently AI (Data Drift & Target Drift)
    logger.info("Running Evidently AI...")
    
    # Drop predictions so they don't pollute drift reports
    ref_evidently = ref_df.drop(columns=['y_pred', 'y_pred_proba'], errors='ignore')
    curr_evidently = curr_df.drop(columns=['y_pred', 'y_pred_proba'], errors='ignore')
    
    metrics = [DataDriftPreset()]
    cm = ColumnMapping()
    
    if has_ground_truth:
        metrics.append(TargetDriftPreset())
        cm.target = 'churn'
    else:
        # If no ground truth, drop 'churn' from reference so Evidently compares only input features
        ref_evidently = ref_evidently.drop(columns=['churn'], errors='ignore')

    drift_report = Report(metrics=metrics)
    drift_report.run(reference_data=ref_evidently, current_data=curr_evidently, column_mapping=cm)
    
    drift_report_path = Path(report_dir) / f"evidently_report_{batch_name}_{timestamp}.html"
    drift_report.save_html(str(drift_report_path))
    logger.info(f"Saved Evidently Report to {drift_report_path}")

    # Cleanly extract Evidently metrics using next()
    drift_dict = drift_report.as_dict()
    data_drift_metric = next(m for m in drift_dict['metrics'] if m['metric'] == 'DatasetDriftMetric')
    drift_percentage = data_drift_metric['result']['share_of_drifted_columns'] * 100
    logger.info(f"Evidently Data Drift: {drift_percentage:.1f}%")

    target_drift_detected = False
    if has_ground_truth:
        target_metric = next(m for m in drift_dict['metrics'] if m['metric'] == 'ColumnDriftMetric' and m['result'].get('column_name') == 'churn')
        target_drift_detected = target_metric['result']['drift_detected']
        logger.info(f"Evidently Target Drift: {'DETECTED' if target_drift_detected else 'NOT detected'}")

    # 4. Performance (Actual vs Estimated)
    baseline_f1 = get_baseline_f1()
    logger.info(f"Baseline F1-Score: {baseline_f1:.4f}")

    if has_ground_truth:
        logger.info("Calculating ACTUAL performance...")
        current_f1 = f1_score(curr_df['churn'], curr_df['y_pred'])
        logger.info(f"ACTUAL F1-Score: {current_f1:.4f}")
    else:
        logger.info("Estimating performance with NannyML CBPE...")
        cbpe = nml.CBPE(
            y_pred_proba='y_pred_proba',
            y_pred='y_pred',
            y_true='churn',
            metrics=['f1'],
            chunk_size=len(curr_df),
            problem_type='classification_binary'
        )
        cbpe.fit(ref_df)
        est_results = cbpe.estimate(curr_df)
        
        current_f1 = est_results.filter(period='analysis').to_df().iloc[0][('f1', 'value')]
        logger.info(f"ESTIMATED F1-Score: {current_f1:.4f}")
        
        nannyml_plot_path = Path(report_dir) / f"nannyml_report_{batch_name}_{timestamp}.html"
        est_results.plot().write_html(str(nannyml_plot_path))
        logger.info(f"Saved NannyML Plot to {nannyml_plot_path}")

    f1_drop_relative = ((baseline_f1 - current_f1) / baseline_f1) * 100
    if f1_drop_relative < 0:
        logger.info(f"📈 Relative F1 Change: Improved by {abs(f1_drop_relative):.1f}%")
    else:
        logger.info(f"📉 Relative F1 Change: Dropped by {f1_drop_relative:.1f}%")

    # 5. Decision Engine
    reasons = []
    
    if drift_percentage > drift_threshold:
        reasons.append(f"Data Drift ({drift_percentage:.1f}%) > {drift_threshold}%")
        
    if f1_drop_relative > f1_drop_threshold:
        metric_type = "Actual" if has_ground_truth else "Estimated"
        reasons.append(f"{metric_type} F1 Drop ({f1_drop_relative:.1f}%) > {f1_drop_threshold}%")
        
    if has_ground_truth and target_drift_detected:
        reasons.append("Target Drift detected")

    retrain_required = len(reasons) > 0

    monitoring_execution_time = time.time() - start_time
    
    # Log metrics to MLflow for Thesis RQ tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("bank_churn_prediction")
    with mlflow.start_run(run_name="monitoring_job"):
        mlflow.log_metric("monitoring_execution_time", monitoring_execution_time)
        mlflow.log_metric("data_drift_percentage", drift_percentage)
        if has_ground_truth:
            mlflow.log_metric("actual_f1", current_f1)
        else:
            mlflow.log_metric("estimated_f1", current_f1)

    # Generate JSON Signal
    signal = {
        "retrain_required": retrain_required,
        "has_ground_truth": bool(has_ground_truth),
        "data_drift_percentage": round(drift_percentage, 2),
        "f1_drop_relative": round(f1_drop_relative, 2),
        "target_drift_detected": bool(target_drift_detected),
        "reasons": reasons,
        "timestamp": timestamp
    }

    signal_path = Path(report_dir) / "signal.json"
    with open(signal_path, "w") as f:
        json.dump(signal, f, indent=4)
        
    logger.info(f"Saved JSON signal to {signal_path}")

    if retrain_required:
        logger.warning("⚠️ RETRAINING SIGNALED!")
        for r in reasons:
            logger.warning(f"Reason: {r}")
        logger.info("Exiting with code 0 (Monitor job completed successfully).")
    else:
        logger.info("✅ Pipeline Healthy. Exiting with code 0.")
        
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str, default="data/reference_with_predictions.csv")
    parser.add_argument("--current", type=str, default="data/batch_1.csv")
    parser.add_argument("--report-dir", type=str, default="reports/")
    parser.add_argument("--drift-threshold", type=float, default=30.0)
    parser.add_argument("--f1-drop-threshold", type=float, default=10.0)
    args = parser.parse_args()
    
    run_monitoring(args.ref, args.current, args.report_dir, args.drift_threshold, args.f1_drop_threshold)
