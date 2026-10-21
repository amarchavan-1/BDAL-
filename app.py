import findspark
findspark.init()

import streamlit as st
import pandas as pd
import plotly.express as px
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Page Configuration - White Minimalist Theme
st.set_page_config(
    page_title="UPI Fraud Analytics Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Custom White Theme CSS
st.markdown("""
    <style>
    /* White Background & Minimalist Styling */
    .stApp {
        background-color: #FFFFFF;
        color: #111827;
    }
    
    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        padding: 16px;
        border-radius: 8px;
        box-shadow: 0px 1px 2px rgba(0, 0, 0, 0.05);
    }
    div[data-testid="stMetric"] label {
        color: #4B5563 !important;
        font-size: 0.875rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #111827 !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }

    /* Table & Container Tweaks */
    .stDataFrame {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_spark_session():
    return (SparkSession.builder
            .appName("Streamlit_UPI_Dashboard")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate())

@st.cache_data
def load_data():
    spark = get_spark_session()
    BASE = "hdfs://localhost:9000/bigdata_project"
    
    df = spark.read.parquet(f"{BASE}/processed/paysim_predictions")
    
    total_txns = df.count()
    actual_fraud = df.filter("isFraud = 1").count()
    predicted_fraud = df.filter("prediction = 1.0").count()
    tp = df.filter("isFraud = 1 AND prediction = 1.0").count()
    fp = df.filter("isFraud = 0 AND prediction = 1.0").count()
    
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
    recall = (tp / actual_fraud) * 100 if actual_fraud > 0 else 0
    
    type_df = df.groupBy("type").agg(
        F.count("*").alias("total"),
        F.sum("isFraud").alias("fraud_count")
    ).toPandas()
    type_df["fraud_pct"] = (type_df["fraud_count"] / type_df["total"]) * 100
    
    top_flagged = df.filter("prediction = 1.0").orderBy(F.desc("amount")).limit(10).toPandas()
    
    return {
        "total_txns": total_txns,
        "actual_fraud": actual_fraud,
        "predicted_fraud": predicted_fraud,
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "type_df": type_df,
        "top_flagged": top_flagged
    }

st.title("🛡️ UPI Fraud Detection Analytics")
st.caption("Real-Time Apache Spark & HDFS Prediction Layer | PCCoE Pune Final Year Project")
st.divider()

with st.spinner("Connecting to HDFS & Spark Cluster..."):
    data = load_data()

# Metric Cards Row (White Cards with Slate Text)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Evaluated Transactions", f"{data['total_txns']:,}")
col2.metric("Actual Ground Frauds", f"{data['actual_fraud']:,}")
col3.metric("Model Flagged Risk", f"{data['predicted_fraud']:,}")
col4.metric("Model Precision", f"{data['precision']}%")
col5.metric("Model Recall", f"{data['recall']}%")

st.divider()

# Charts Row - Clean White Plotly Templates
c1, c2 = st.columns(2)

with c1:
    st.markdown("##### **Fraud Rate by Transaction Type (%)**")
    fig_bar = px.bar(
        data['type_df'], 
        x='type', 
        y='fraud_pct', 
        text_auto='.2f',
        labels={'type': 'Transaction Type', 'fraud_pct': 'Fraud Rate (%)'},
        color_discrete_sequence=['#DC2626'] # Clean Crimson Red
    )
    fig_bar.update_layout(
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with c2:
    st.markdown("##### **Fraud Prediction Split**")
    labels = ['Flagged Risk', 'Genuine Transactions']
    values = [data['predicted_fraud'], data['total_txns'] - data['predicted_fraud']]
    fig_pie = px.pie(
        names=labels, 
        values=values, 
        hole=0.6, 
        color_discrete_sequence=['#DC2626', '#2563EB'] # Red vs Slate Blue
    )
    fig_pie.update_layout(
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# Data Table
st.markdown("##### **Top 10 High-Risk Transactions Flagged by Random Forest**")
st.dataframe(
    data['top_flagged'][['step', 'type', 'amount', 'nameOrig', 'nameDest', 'isFraud', 'prediction']],
    use_container_width=True
)
