from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (SparkSession.builder.appName("UPI_Fraud_Analysis")
         .config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

BASE = "hdfs://localhost:9000/bigdata_project"

# 1. Load raw dataset from HDFS
df = spark.read.csv(f"{BASE}/raw/paysim/paysim.csv", header=True, inferSchema=True)

# 2. Clean and add temporal features
df = (df.dropna()
        .withColumn("hour_of_day", F.col("step") % 24)
        .withColumn("day", (F.col("step") / 24).cast("int") + 1)
        .withColumn("day_of_week", (F.col("day") - 1) % 7))

# 3. Write Parquet output to HDFS
df.write.mode("overwrite").parquet(f"{BASE}/processed/paysim_clean")
print("Parquet written successfully.")

# 4. Read processed Parquet for analysis
df = spark.read.parquet(f"{BASE}/processed/paysim_clean")
print("Clean rows count:", df.count())

print("\n=== Analysis A: Fraud rate by transaction type ===")
(df.groupBy("type")
   .agg(F.count("*").alias("total"),
        F.sum("isFraud").alias("fraud_count"),
        F.round(F.avg("isFraud") * 100, 4).alias("fraud_pct"))
   .orderBy(F.desc("fraud_pct"))
   .show())

print("\n=== Analysis B: Fraud by hour of day ===")
(df.groupBy("hour_of_day")
   .agg(F.count("*").alias("total"),
        F.sum("isFraud").alias("fraud_count"),
        F.round(F.avg("isFraud") * 100, 4).alias("fraud_pct"))
   .orderBy("hour_of_day")
   .show(24))

print("\n=== Analysis C: Amount statistics: fraud vs genuine ===")
(df.groupBy("isFraud")
   .agg(F.count("*").alias("n"),
        F.round(F.avg("amount"), 2).alias("avg_amount"),
        F.round(F.max("amount"), 2).alias("max_amount"))
   .show())

print("\n=== Analysis D: Top 10 sender accounts by fraud count ===")
(df.filter("isFraud = 1").groupBy("nameOrig")
   .agg(F.count("*").alias("fraud_txns"), F.round(F.sum("amount"), 2).alias("total_fraud_amount"))
   .orderBy(F.desc("fraud_txns"))
   .show(10, truncate=False))

spark.stop()
