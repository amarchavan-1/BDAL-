from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

spark = (SparkSession.builder.appName("UPI_Fraud_Save_Predictions")
         .config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

BASE = "hdfs://localhost:9000/bigdata_project"

# 1. Load Parquet
df = spark.read.parquet(f"{BASE}/processed/paysim_clean")
df_model = df.filter(F.col("type").isin(["TRANSFER", "CASH_OUT"]))

# 2. Vector Assembly
indexer = StringIndexer(inputCol="type", outputCol="type_index")
df_indexed = indexer.fit(df_model).transform(df_model)

feature_cols = ["type_index", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "hour_of_day"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(df_indexed)

# 3. Fit Random Forest & Generate Predictions
rf = RandomForestClassifier(labelCol="isFraud", featuresCol="features", numTrees=20, maxDepth=5, seed=42)
model = rf.fit(data)
predictions = model.transform(data)

# 4. Save high-risk predictions to HDFS as Parquet
output_df = predictions.select("step", "type", "amount", "nameOrig", "nameDest", "isFraud", "prediction")
output_df.write.mode("overwrite").parquet(f"{BASE}/processed/paysim_predictions")

print("Predictions saved successfully to HDFS.")
spark.stop()
