import os
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize Spark
spark = SparkSession.builder.appName("UPI_Fraud_Visualizations").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

BASE = "hdfs://localhost:9000/bigdata_project"
output_dir = os.path.expanduser("~/bigdata_project/output/charts")
os.makedirs(output_dir, exist_ok=True)

# Load Processed Parquet Data
df = spark.read.parquet(f"{BASE}/processed/paysim_clean")

# ---------------------------------------------------------
# 1. Heatmap: Fraud Rate by Hour of Day & Day of Week
# ---------------------------------------------------------
print("Generating Heatmap data...")
heatmap_df = (df.groupBy("day_of_week", "hour_of_day")
                .agg(F.mean("isFraud").alias("fraud_rate"))
                .toPandas())

heatmap_pivot = heatmap_df.pivot(index="day_of_week", columns="hour_of_day", values="fraud_rate")

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_pivot, cmap="YlOrRd", annot=False, cbar_kws={'label': 'Fraud Rate'})
plt.title("Advanced Visualization: Fraud Density Heatmap (Hour vs Day of Week)")
plt.xlabel("Hour of Day (0 - 23)")
plt.ylabel("Day of Week")
plt.tight_layout()
plt.savefig(f"{output_dir}/1_fraud_heatmap.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# 2. Time-Series: Daily Transaction Volume & Fraud Trend
# ---------------------------------------------------------
print("Generating Time-Series data...")
timeseries_df = (df.groupBy("day")
                   .agg(F.count("*").alias("total_txns"), F.sum("isFraud").alias("fraud_txns"))
                   .orderBy("day")
                   .toPandas())

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

ax1.plot(timeseries_df["day"], timeseries_df["total_txns"], color="blue", label="Total Volume", linewidth=2)
ax2.plot(timeseries_df["day"], timeseries_df["fraud_txns"], color="red", label="Fraud Volume", linestyle="--", linewidth=2)

ax1.set_xlabel("Day (Step / 24)")
ax1.set_ylabel("Total Transactions", color="blue")
ax2.set_ylabel("Fraud Transactions", color="red")
plt.title("Time-Series Visualization: Transaction Volume vs Fraud Over Time")
plt.tight_layout()
plt.savefig(f"{output_dir}/2_time_series.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# 3. Categorical Bar Plot: Fraud Rate by Transaction Type
# ---------------------------------------------------------
print("Generating Transaction Type chart...")
type_df = (df.groupBy("type")
             .agg((F.avg("isFraud") * 100).alias("fraud_pct"))
             .orderBy(F.desc("fraud_pct"))
             .toPandas())

plt.figure(figsize=(8, 5))
ax = sns.barplot(x="type", y="fraud_pct", data=type_df, palette="Reds_r")
plt.title("Fraud Percentage by Transaction Type")
plt.xlabel("Transaction Type")
plt.ylabel("Fraud Percentage (%)")

for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig(f"{output_dir}/3_transaction_types.png", dpi=300)
plt.close()

print(f"\nAll visualization plots successfully saved to: {output_dir}")
spark.stop()
