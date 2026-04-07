# Bank Churn MLOps Pipeline

This repository contains a lightweight, fully automated MLOps pipeline for predicting bank customer churn. It is designed to demonstrate end-to-end machine learning operations, including data validation, drift detection, performance estimation, automated retraining, and shadow deployment.

---

## 🏗️ System Architecture & Logic

The pipeline is built with strict **modularity** and **statelessness**. It is orchestrated via GitHub Actions and uses MLflow as its central memory (Model Registry and Metric Tracking).

The automated flow works as follows:
1. **Validation (`src/validation.py`)**: The Gatekeeper. Blocks structurally corrupted data.
2. **Monitoring (`src/monitor.py`)**: The Observer. Detects drift and performance decay. Outputs a `signal.json` decision file.
3. **Retraining (`src/train.py`)**: The Challenger Creator. If triggered by the JSON signal, it trains a new model on the drifted data and registers it in MLflow as `staging`.
4. **Registry (`src/registry.py`)**: The Judge. Pits the `staging` model against the `production` model on an unseen holdout set. Promotes the winner.

---

## 🧪 How to Test the Pipeline (Thesis Scenarios)

To demonstrate the full capabilities of this MLOps pipeline, we have prepared specific data batches that simulate different real-world scenarios.

### 1. The "Happy Path" (Healthy Pipeline)
* **Goal:** Prove the pipeline works when data is normal.
* **File to use:** `data/batch_1.csv`
* **Expected Result:** Validation passes. Monitoring detects 0% drift. `signal.json` reports no retraining needed.

### 2. The "Gatekeeper" (Data Validation Failure)
* **Goal:** Prove the pipeline blocks corrupted data before it breaks the monitoring tools.
* **Files to use:** `data/batch_1_broken_data.csv` or `data/batch_1_semantic_errors.csv`
* **Expected Result:** Validation catches the errors, logs a 🚨 critical failure, and immediately halts the pipeline (Exit Code 1).

### 3. Data Drift (The "Inflation" Scenario)
* **Goal:** Prove the system can detect when the input features (X) change drastically, and estimate the performance drop *before* the true labels arrive.
* **File to use:** `data/batch_2_drifted_unlabeled.csv` (Balances and salaries multiplied by 1.5x).
* **Expected Result:** Evidently detects Data Drift. NannyML estimates a massive drop in the F1-Score. The pipeline signals for retraining, but gracefully skips it because ground truth labels are missing.

### 4. Target Drift (The "Economic Crash" Scenario)
* **Goal:** Prove the system can detect when the fundamental behavior of the customers (Y) changes, and successfully execute a Shadow Deployment.
* **File to use:** `data/batch_3_target_drift.csv` (Massive sudden churn).
* **Expected Result:** Evidently detects Target Drift. The system calculates the *Actual* F1 drop, triggers retraining, trains a Challenger model, pits it against the Champion, and promotes the Challenger to Production.

### 5. Concept Drift (The "Blind Spot" Scenario)
* **Goal:** Prove the mathematical limitations of Performance Estimation algorithms.
* **File to use:** `data/batch_4_concept_drift_unlabeled.csv` (Young people suddenly churn, but input features look normal).
* **Expected Result:** NannyML is "blind" to this pure concept drift and incorrectly estimates that the F1-Score is healthy. *This is a fundamental limitation of all estimation algorithms and a crucial point for thesis discussion!*

---

## 🛠️ Component Breakdown

### 1. `src/validation.py` - The Data Gatekeeper
Acts as a strict bouncer. 
* **Layer 1 (Pandas):** Instantly verifies column counts and data types to prevent deep analytical tools from crashing.
* **Layer 2 (Evidently AI):** Scans for missing values and semantic business logic (e.g., age ranges). 
* **Logic:** "Fail-Fast". If any check fails, the pipeline dies immediately (Exit Code 1) to protect the system.

### 2. `src/monitor.py` - The Brain
Detects if the world has changed. Dynamically adapts based on whether we have the true answers (ground truth) yet.
* **If Answers MISSING:** Uses **Evidently AI** for Data Drift and **NannyML (CBPE)** to mathematically estimate the F1-Score drop.
* **If Answers PRESENT:** Calculates the *Actual* F1-Score and uses **Evidently AI** to detect Target Drift.
* **Output:** Logs metrics to MLflow and generates `reports/signal.json` (e.g., `{"retrain_required": true}`). It exits with Code 0 so CI/CD pipelines don't crash, allowing the orchestrator to read the JSON and decide what to do next.

### 3. `src/train.py` - Baseline & Challenger Training
Handles both initial model creation and automated retraining. Uses a Scikit-Learn Pipeline (StandardScaler, OneHotEncoder, RandomForest).
* **Baseline Mode:** Trains on the 6,000-row historical dataset. Registers the model in MLflow with the `production` alias.
* **Retrain Mode (Shadow Deployment):** When drift occurs, it *discards* historical data to avoid the "Poisoned Data" problem. It splits the 1,000 new drifted rows (800 Train / 200 Test). It trains a new model purely on the 800 rows, registers it in MLflow with the `staging` alias, and saves the 200 unseen rows as `staging_test_set.csv` for the Judge.

### 4. `src/registry.py` - The Judge (Promotion Logic)
Executes the final step of the Shadow Deployment.
* **The Battle:** It loads the Champion (`@production`) and the Challenger (`@staging`) from MLflow. It asks both to predict on the exact same 200-row unseen holdout set (`staging_test_set.csv`).
* **The Decision:** If the Challenger's F1-score is higher, it officially moves the `production` alias to the Challenger. It then overwrites the monitoring baseline so `monitor.py` accepts the new data as the "new normal".

---

## 📊 MLflow & Thesis Metrics

The pipeline automatically logs specific metrics to MLflow to answer the core Research Questions (RQs) of the thesis:
* **RQ1 (Detection Accuracy):** `monitor.py` logs `estimated_f1` and `actual_f1` to the `monitoring_job` run.
* **RQ2 (Reaction Speed):** `monitor.py` logs `monitoring_execution_time`, and `train.py` logs `retraining_execution_time`.
* **RQ3 (Recovery Rate):** `registry.py` logs `production_f1` (Before), `challenger_f1` (After), and `f1_improvement` to the `model_promotion_evaluation` run.

---

## 🚀 How to Run the Pipeline

### Option A: Run in the Cloud (GitHub Actions)
This is the recommended, fully automated orchestration method.
1. Push this repository to GitHub.
2. Go to the **Actions** tab.
3. Select **Bank Churn MLOps Pipeline** and click **Run workflow**.
4. Enter the batch file you want to test (e.g., `data/batch_3_target_drift.csv`).
5. Download the `mlops-artifacts-final.zip` at the end of the run to view your HTML reports and MLflow metrics.

### Option B: Run Locally (Manual Orchestration)
**Mac Users Note:** You must install the OpenMP runtime first because NannyML uses XGBoost: `brew install libomp`

**1. Reset the System to Day 0**
We built a utility script to instantly wipe all history, recreate the datasets, and train the baseline model:
```bash
python3 reset.py
```

**2. Step-by-Step Execution**
```bash
# 1. Run Validation
python3 src/validation.py --ref data/reference.csv --current data/batch_3_target_drift.csv

# 2. Run Monitoring
python3 src/monitor.py --ref data/reference_with_predictions.csv --current data/batch_3_target_drift.csv

# 3. Run Retraining (If signal.json says true)
python3 src/train.py --retrain data/batch_3_target_drift.csv

# 4. Evaluate and Promote
python3 src/registry.py

# 5. View Metrics
mlflow ui
```