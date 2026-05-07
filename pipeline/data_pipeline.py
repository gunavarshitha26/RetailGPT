"""
============================================
RetailGPT — Data Engineering Pipeline
============================================
Local Emulation of an Azure Data Platform:
- Stage 1: Raw Ingestion (Azure Data Factory)
- Stage 2: Staged Processing (Azure Databricks)
- Stage 3: Curated Analytics (Microsoft Fabric)
"""

import os
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lightgbm import LGBMRegressor
from sklearn.ensemble import IsolationForest
from mlxtend.frequent_patterns import apriori, association_rules

from backend.config import settings

def stage1_raw_ingestion(source_file: str) -> str:
    """
    [AZURE DATA FACTORY EQUIVALENT]
    Ingests raw data from source (e.g., Blob Storage) to Raw Zone.
    """
    print("🚀 STAGE 1: RAW INGESTION...")
    df = pd.read_csv(source_file)
    
    # Basic validation
    expected_cols = [
        "Row ID", "Order ID", "Order Date", "Ship Date", "Ship Mode", 
        "Customer ID", "Customer Name", "Segment", "Country", "City", 
        "State", "Postal Code", "Region", "Product ID", "Category", 
        "Sub-Category", "Product Name", "Sales"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw data: {missing}")
        
    raw_path = os.path.join(settings.RAW_DATA_DIR, "train_raw.parquet")
    df.to_parquet(raw_path, index=False)
    print(f"✅ Saved Raw: {raw_path}")
    return raw_path


def stage2_staged_processing(raw_path: str) -> dict:
    """
    [AZURE DATABRICKS EQUIVALENT (PySpark Logic Emulated in Pandas)]
    Cleans dates, standardizes types, and creates base aggregations.
    """
    print("\n🚀 STAGE 2: STAGED PROCESSING...")
    df = pd.read_parquet(raw_path)
    
    # Clean Dates (Strip timezones, enforce DD/MM/YYYY)
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y').dt.tz_localize(None)
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y').dt.tz_localize(None)
    
    # Base Staged Data
    staged_path = os.path.join(settings.STAGED_DATA_DIR, "train_staged.parquet")
    df.to_parquet(staged_path, index=False)
    
    # Aggregation 1: Daily Sales
    daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()
    daily_sales = daily_sales.sort_values('Order Date')
    
    # Aggregation 2: Category Sales by Month
    df['Month_Year'] = df['Order Date'].dt.to_period('M').dt.to_timestamp()
    category_sales = df.groupby(['Month_Year', 'Category'])['Sales'].sum().reset_index()
    
    # Aggregation 3: Segment x Region
    segment_region = df.groupby(['Region', 'Segment']).agg(
        Total_Sales=('Sales', 'sum'),
        Order_Count=('Order ID', 'nunique')
    ).reset_index()
    segment_region['Avg_Order_Value'] = segment_region['Total_Sales'] / segment_region['Order_Count']
    
    # Save Staged Aggregations
    daily_path = os.path.join(settings.STAGED_DATA_DIR, "agg_daily_sales.parquet")
    cat_path = os.path.join(settings.STAGED_DATA_DIR, "agg_category_sales.parquet")
    seg_path = os.path.join(settings.STAGED_DATA_DIR, "agg_segment_region.parquet")
    
    daily_sales.to_parquet(daily_path, index=False)
    category_sales.to_parquet(cat_path, index=False)
    segment_region.to_parquet(seg_path, index=False)
    
    print(f"✅ Saved Staged Aggregations.")
    return {
        "staged_base": staged_path,
        "daily_sales": daily_path,
        "category_sales": cat_path,
        "segment_region": seg_path
    }


def stage3_curated_analytics(staged_paths: dict):
    """
    [MICROSOFT FABRIC / SYNAPSE EQUIVALENT]
    Runs ML Models (LightGBM, Isolation Forest) and advanced analytics.
    Produces final curated files ready for App / Power BI consumption.
    """
    print("\n🚀 STAGE 3: CURATED ANALYTICS (ML & MINING)...")
    
    # ---------------------------------------------------------
    # A. Forecast & Anomalies (LightGBM + Isolation Forest)
    # ---------------------------------------------------------
    print("   -> Running ML Engine...")
    daily_df = pd.read_parquet(staged_paths["daily_sales"])
    
    # Feature Engineering for LightGBM
    daily_df['TimeIndex'] = daily_df['Order Date'].map(pd.Timestamp.toordinal)
    daily_df['DayOfWeek'] = daily_df['Order Date'].dt.dayofweek
    daily_df['Month'] = daily_df['Order Date'].dt.month
    
    # Optional rolling features if enough data
    daily_df['Rolling_7d'] = daily_df['Sales'].rolling(7, min_periods=1).mean()
    
    X = daily_df[['TimeIndex', 'DayOfWeek', 'Month', 'Rolling_7d']]
    y = daily_df['Sales']
    
    # LightGBM Forecast
    model = LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    daily_df['Forecast'] = model.predict(X)
    
    # Isolation Forest Anomaly Detection
    iso = IsolationForest(contamination=0.05, random_state=42)
    daily_df['Anomaly_Flag'] = iso.fit_predict(daily_df[['Sales']]) # -1 = Anomaly, 1 = Normal
    
    # Smart Anomaly Classification
    def classify_anomaly(row):
        if row['Anomaly_Flag'] == 1:
            return "Normal"
        if row['Sales'] > row['Rolling_7d']:
            return "Positive (Sales Spike)"
        return "Negative (Sales Drop)"
        
    daily_df['Anomaly_Type'] = daily_df.apply(classify_anomaly, axis=1)
    
    # Generate actionable advice based on playbook
    def generate_advice(row):
        if row['Anomaly_Type'] == "Normal":
            return ""
        if row['Anomaly_Type'] == "Positive (Sales Spike)":
            return "Check inventory levels. Expedite POs for high-velocity items. Prevent stockouts."
        return "Investigate localized drops. Consider bundling slow movers or targeted segment promotions."
        
    daily_df['Actionable_Advice'] = daily_df.apply(generate_advice, axis=1)
    
    # ---------------------------------------------------------
    # B. Market Basket Analysis (Apriori)
    # ---------------------------------------------------------
    print("   -> Running Market Basket Analysis...")
    base_df = pd.read_parquet(staged_paths["staged_base"])
    
    # Group by Order ID and create a list of Sub-Categories
    basket = base_df.groupby(['Order ID', 'Sub-Category'])['Sales'].sum().unstack().reset_index().fillna(0).set_index('Order ID')
    basket = basket.apply(lambda x: x > 0) # Convert to boolean
    
    # Run Apriori
    frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        rules = rules.sort_values('lift', ascending=False).head(50)
        
        # Clean up frozensets for Parquet serialization
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        basket_path = os.path.join(settings.CURATED_DATA_DIR, "market_basket_rules.parquet")
        rules.to_parquet(basket_path, index=False)
        print("      ✅ Market Basket rules generated.")
    else:
        print("      ⚠️ No frequent itemsets found with min_support=0.01")
        
    # ---------------------------------------------------------
    # C. Save Curated Final Files
    # ---------------------------------------------------------
    forecast_path = os.path.join(settings.CURATED_DATA_DIR, "forecast_anomalies.parquet")
    daily_df.to_parquet(forecast_path, index=False)
    
    # Copy staged aggregations directly to curated
    cat_df = pd.read_parquet(staged_paths["category_sales"])
    cat_df.to_parquet(os.path.join(settings.CURATED_DATA_DIR, "category_timeseries.parquet"), index=False)
    
    seg_df = pd.read_parquet(staged_paths["segment_region"])
    seg_df.to_parquet(os.path.join(settings.CURATED_DATA_DIR, "segment_region.parquet"), index=False)
    
    print("\n🎉 PIPELINE COMPLETE! Curated data is ready in /curated.")


def run_pipeline(source_csv_path: str):
    """Executes the full pipeline end-to-end."""
    from pipeline.__init__ import ensure_directories
    ensure_directories()
    
    raw_file = stage1_raw_ingestion(source_csv_path)
    staged_files = stage2_staged_processing(raw_file)
    stage3_curated_analytics(staged_files)

if __name__ == "__main__":
    # Local execution for testing
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_pipeline("train.csv")
