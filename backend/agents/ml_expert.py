import os
import pandas as pd
from backend.config import settings

def get_ml_expert_response(query: str) -> str:
    """
    ML Expert Agent.
    Reads the curated forecast/anomalies Parquet files and explains them.
    """
    from backend.agents import get_llm
    
    forecast_path = os.path.join(settings.CURATED_DATA_DIR, "forecast_anomalies.parquet")
    if not os.path.exists(forecast_path):
        return "The ML Pipeline has not generated a forecast yet."
        
    try:
        df = pd.read_parquet(forecast_path)
        anomalies = df[df['Anomaly_Type'] != "Normal"]
        pos_anomalies = len(df[df['Anomaly_Type'] == "Positive (Sales Spike)"])
        neg_anomalies = len(df[df['Anomaly_Type'] == "Negative (Sales Drop)"])
        
        stats = f"""
        ML Pipeline Status: Active
        Total Days Forecasted: {len(df)}
        Detected Positive Anomalies (Spikes): {pos_anomalies}
        Detected Negative Anomalies (Drops): {neg_anomalies}
        """
        
        llm = get_llm()
        if llm:
            prompt = f"""You are the Lead Machine Learning Engineer for RetailGPT.
            You use LightGBM for forecasting and Isolation Forest for anomaly detection.
            Here are the current pipeline statistics:
            {stats}
            
            Answer the user's question expertly.
            
            Question: {query}
            """
            response = llm.invoke(prompt)
            return response.content
        else:
            return f"[Offline Mode] ML Expert Status:\n{stats}"
            
    except Exception as e:
        return f"Error analyzing ML results: {str(e)}"
