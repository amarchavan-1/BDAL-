# 🛡️ Big Data Financial Fraud Detection & Risk Analytics Platform

An enterprise-grade, distributed Big Data pipeline engineered to process, clean, analyze, and detect real-time financial fraud across multi-million payment transactions. Built on **Apache Hadoop (HDFS)**, **Apache Spark (PySpark MLlib)**, and **Streamlit**, this project classifies fraudulent transaction patterns across high-volume digital banking channels with a **99.47% Precision Rate**.

Developed as a Capstone Mini-Project at **Pimpri Chinchwad College of Engineering (PCCoE), Pune** (Department of Information Technology, Academic Year 2026–27).

---

## 👥 Project Team & Governance

- **Project Team:**
  - Shreyas Lohakare
  - Sagar Najardhane
  - Amar Chavan
- **Project Guide:** Dr. Jaya Dewan
- **Institution:** Pimpri Chinchwad College of Engineering (PCCoE), Pune

---

## 📌 Executive Summary & Key Results

| Metric / Attribute             | Value         | Description                                            |
| :----------------------------- | :------------ | :----------------------------------------------------- |
| **Total Ingested Volume**      | **6,362,620** | Raw PaySim transactional records (~500 MB CSV)         |
| **Storage Compression**        | **~266 MB**   | Columnar Snappy-compressed Apache Parquet on HDFS      |
| **Evaluated High-Risk Volume** | **2,770,409** | Filtered `TRANSFER` & `CASH_OUT` transaction subset    |
| **Ground-Truth Frauds**        | **8,213**     | Confirmed fraud cases (100% in Transfer/Cash-Out)      |
| **Model Precision**            | **99.47%**    | True positive rate among flagged high-risk alerts      |
| **Model F1 Score**             | **0.9985**    | Overall classification balance score                   |
| **ROC AUC Score**              | **0.9434**    | Area under the Receiver Operating Characteristic curve |

---

## 🏗️ System Architecture

[ Raw PaySim Dataset (~500MB CSV) ]
│
▼
┌──────────────────────────────────────────────┐
│ Hadoop Distributed File System (HDFS) │
│ Path: /bigdata_project/raw/ │
└──────────────────────┬───────────────────────┘
│
▼
┌──────────────────────────────────────────────┐
│ Apache Spark / PySpark Ingestion Engine │
│ • Explicit Schema Enforcement & Null Removal │
│ • Non-Risk Filtering (Payment/CashIn/Debit) │
│ • Feature Engineering (Balance Differentials)│
└──────────────────────┬───────────────────────┘
│
▼
┌──────────────────────────────────────────────┐
│ PySpark MLlib Machine Learning Pipeline │
│ • VectorAssembler Feature Pipeline │
│ • Random Forest Classifier (20 Trees) │
│ • Model Evaluation & Prediction Generation │
└──────────────────────┬───────────────────────┘
│
▼
┌──────────────────────────────────────────────┐
│ HDFS Columnar Persistence │
│ Path: /bigdata_project/processed/ │
│ Format: Snappy Parquet │
└──────────────────────┬───────────────────────┘
│
▼
┌──────────────────────────────────────────────┐
│ Interactive Streamlit Web Analytics App │
│ • FindSpark Context Link to PySpark Engine │
│ • Plotly Interactive Visuals & Metric Cards │
│ • URL: http://localhost:8501 │
└──────────────────────────────────────────────┘

---

## 📂 Project Directory Structure

```text
bigdata_project/
├── data/                         # Local raw data landing zone
├── scripts/                      # PySpark ETL, ML, and evaluation scripts
│   ├── 01_analysis.py            # HDFS ingestion, cleaning & aggregations
│   ├── 02_ml_model.py            # PySpark MLlib Random Forest model training
│   └── 05_dashboard_summary.py   # Prediction metric evaluator & terminal parser
├── output/                       # Visual assets & chart output PNGs
│   └── charts/
│       ├── 1_fraud_heatmap.png   # 24-Hour fraud density heatmap
│       ├── 2_time_series.png     # 30-Day transaction volume vs. fraud trend
│       └── 3_transaction_types.png # Transaction type fraud percentage plot
├── logs/                         # Execution logs from YARN/Spark cluster runs
├── app.py                        # Streamlit white-themed interactive dashboard
└── README.md                     # Comprehensive project documentation
⚙️ Prerequisites & Infrastructure Stack
Operating Environment: WSL2 (Ubuntu 24.04 LTS on Windows 11)

Core Distributed Technologies:

Apache Hadoop 3.3.x (HDFS NameNode & DataNode)

Apache Spark 3.5.3 (PySpark MLlib)

Python Runtime: Python 3.14 / 3.10+

Dependencies: pyspark, findspark, streamlit, plotly, pandas

🚀 Setup & Execution Manual
Step 1: Initialize Project Directory Structure
Bash
# Create directory tree
mkdir -p ~/bigdata_project/{data,scripts,output/charts,logs}
cd ~/bigdata_project
Step 2: Install Python Dependencies
Bash
# Install required libraries
python3 -m pip install findspark streamlit plotly pandas --no-cache-dir --break-system-packages

# Add local path to environment
export PATH="$HOME/.local/bin:$PATH"
Step 3: Start Hadoop HDFS Services
Bash
# Start Hadoop HDFS daemons
start-dfs.sh

# Confirm running daemons (NameNode, DataNode, SecondaryNameNode)
jps
Step 4: Upload Raw Dataset to HDFS
Bash
# Create HDFS storage folders
hdfs dfs -mkdir -p /bigdata_project/raw
hdfs dfs -mkdir -p /bigdata_project/processed

# Upload raw CSV log to HDFS
hdfs dfs -put ~/PS_20174392719_1491208443941_log.csv /bigdata_project/raw/

# Verify upload
hdfs dfs -ls /bigdata_project/raw/
Step 5: Execute Data Ingestion & Exploratory Analysis
Run 01_analysis.py to enforce schema, clean data, compute aggregations, and write snappy Parquet files to HDFS:

Bash
spark-submit --master local[2] --driver-memory 2g ~/bigdata_project/scripts/01_analysis.py
Step 6: Train Random Forest Machine Learning Model
Run 02_ml_model.py to build feature vectors, train the Random Forest model on 80% data, evaluate metrics on 20% test data, and persist predictions to HDFS:

Bash
spark-submit --master local[2] --driver-memory 2g ~/bigdata_project/scripts/02_ml_model.py
Step 7: View Summary Terminal Metrics
Bash
spark-submit --master local[2] --driver-memory 2g ~/bigdata_project/scripts/05_dashboard_summary.py
📊 Launching the Streamlit Web Application
The interactive dashboard connects directly to your live PySpark session and reads predictions from HDFS Parquet storage.

Bash
# Ensure local executable path is loaded
export PATH="$HOME/.local/bin:$PATH"

# Run Streamlit dashboard
streamlit run ~/bigdata_project/app.py
Navigate to your browser:

Plaintext
http://localhost:8501
```
