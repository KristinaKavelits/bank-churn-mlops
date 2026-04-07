# Bank Churn MLOps Pipeline - Project Rules

## Project Overview
A lightweight, highly modular MLOps pipeline for predicting bank customer churn.
The primary goal is to demonstrate MLOps concepts (Validation, Monitoring, Retraining, Shadow Deployment) for a bachelor's thesis.

## Technical Stack
- **Language**: Python 3.10+
- **ML**: scikit-learn (RandomForestClassifier, StandardScaler, OneHotEncoder)
- **Monitoring**: Evidently AI (DataDriftPreset, TargetDriftPreset), NannyML (CBPE for Performance Estimation)
- **Tracking & Registry**: MLflow (Local tracking, Model Registry)
- **Environment**: Docker, virtual environments (venv)

## Pipeline Stages & Modularity
Each stage must be a standalone Python module, independently testable, using `src/logger.py` (emoji-based logging), Google-style docstrings, and type hinting. Exit codes define the flow (0: healthy, 1: validation error, 2: retraining needed).

1. **Ingestion (`src/ingestion.py`)**: Loads raw data, drops 'customer_id' and 'surname'. Splits into `reference.csv` (5k rows) and batches (`batch_1.csv` to `batch_5.csv`, 1k rows each).
2. **Validation (`src/validation.py`)**: Two-layered "fail-fast" approach. Layer 1: Pandas structural checks. Layer 2: Evidently AI TestSuite for semantic checks. Exits with code 1 on failure.
3. **Monitoring (`src/monitor.py`)**: Dynamically detects Data Drift (Evidently) and/or estimates Performance Drop (NannyML CBPE) based on the presence of ground truth ('churn'). Generates a signal.json to trigger retraining.
4. **Training (`src/train.py`)**: Trains the baseline/retrained model. Logs parameters, metrics (Accuracy, F1, Precision, Recall), and plots (Confusion Matrix, Feature Importance) to MLflow.
5. **Registry & Shadow Deployment (`src/registry.py`)**: *Pending implementation.* Handles MLflow model registry (Staging vs Production) and shadow deployment logic.
6. **Orchestration (`main.py`)**: *Pending implementation.* Orchestrates the end-to-end flow using the exit codes of the individual modules.

## Key Concepts
- **Delayed Ground Truth**: Simulated by dropping the 'churn' column in certain batches.
- **Shadow Deployment**: New models are deployed alongside the production model to compare performance before promotion.
