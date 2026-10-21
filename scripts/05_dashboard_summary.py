from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("UPI_Fraud_Dashboard").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

BASE = "hdfs://localhost:9000/bigdata_project"
df = spark.read.parquet(f"{BASE}/processed/paysim_predictions")

print("\n" + "="*55)
print("       UPI FRAUD DETECTION SYSTEM - SUMMARY DASHBOARD")
print("="*55)

total_txns = df.count()
actual_fraud = df.filter("isFraud = 1").count()
predicted_fraud = df.filter("prediction = 1.0").count()
tp = df.filter("isFraud = 1 AND prediction = 1.0").count()
fp = df.filter("isFraud = 0 AND prediction = 1.0").count()

precision = round((tp / (tp + fp)) * 100, 2) if (tp + fp) > 0 else 0
recall = round((tp / actual_fraud) * 100, 2) if actual_fraud > 0 else 0

print(f"Total Evaluated Transactions : {total_txns:,}")
print(f"Actual Ground-Truth Frauds   : {actual_fraud:,}")
print(f"Model-Flagged High-Risk      : {predicted_fraud:,}")
print(f"Precision                    : {precision}%")
print(f"Recall                       : {recall}%")

print("\n--- TOP HIGH-RISK TRANSACTIONS DETECTED ---")
(df.filter("prediction = 1.0")
   .select("step", "type", "amount", "nameOrig", "nameDest", "isFraud")
   .orderBy(F.desc("amount"))
   .show(10, truncate=False))

spark.stop()
