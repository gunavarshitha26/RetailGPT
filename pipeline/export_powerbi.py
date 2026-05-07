"""
============================================
RetailGPT — Power BI Exporter
============================================
Exports curated Parquet files from Microsoft Fabric (emulated) 
into normalized CSVs optimized for Power BI import.
"""

import os
import pandas as pd
from backend.config import settings

def export_for_powerbi():
    print("🔄 Generating Power BI Export Files...")
    
    # Ensure export directory exists
    os.makedirs(settings.POWERBI_EXPORT_DIR, exist_ok=True)
    
    # 1. Forecast & Anomalies
    forecast_path = os.path.join(settings.CURATED_DATA_DIR, "forecast_anomalies.parquet")
    if os.path.exists(forecast_path):
        df = pd.read_parquet(forecast_path)
        # Rename columns to be more human-readable in Power BI
        df = df.rename(columns={
            "Order Date": "Date",
            "Sales": "Actual Sales",
            "Anomaly_Type": "Anomaly Type",
            "Actionable_Advice": "Anomaly Reason & Advice"
        })
        export_path = os.path.join(settings.POWERBI_EXPORT_DIR, "powerbi_forecast.csv")
        df.to_csv(export_path, index=False)
        print(f"✅ Exported: {export_path}")
        
    # 2. Market Basket
    basket_path = os.path.join(settings.CURATED_DATA_DIR, "market_basket_rules.parquet")
    if os.path.exists(basket_path):
        df = pd.read_parquet(basket_path)
        export_path = os.path.join(settings.POWERBI_EXPORT_DIR, "powerbi_basket.csv")
        df.to_csv(export_path, index=False)
        print(f"✅ Exported: {export_path}")
        
    # 3. Category Sales
    cat_path = os.path.join(settings.CURATED_DATA_DIR, "category_timeseries.parquet")
    if os.path.exists(cat_path):
        df = pd.read_parquet(cat_path)
        df = df.rename(columns={"Month_Year": "Date"})
        export_path = os.path.join(settings.POWERBI_EXPORT_DIR, "powerbi_categories.csv")
        df.to_csv(export_path, index=False)
        print(f"✅ Exported: {export_path}")
        
    # 4. Segment & Region
    seg_path = os.path.join(settings.CURATED_DATA_DIR, "segment_region.parquet")
    if os.path.exists(seg_path):
        df = pd.read_parquet(seg_path)
        export_path = os.path.join(settings.POWERBI_EXPORT_DIR, "powerbi_segments.csv")
        df.to_csv(export_path, index=False)
        print(f"✅ Exported: {export_path}")

    print("🎉 Power BI exports complete. Import the CSVs from 'powerbi_export/' directory into your dashboard.")

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    export_for_powerbi()
