import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from backend.database import save_upload_metadata, get_user_history, add_user_file, get_user_files, delete_user_file_record
from backend.routers.auth import get_current_user_from_cookie
from backend.config import settings

# If pipeline module is not in path (depending on how uvicorn is started)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.data_pipeline import run_pipeline

router = APIRouter(prefix="/api/data", tags=["Data Management"])

@router.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """Uploads a new file and stages it based on type."""
    try:
        user = get_current_user_from_cookie(request)
        username = user["username"]
    except Exception:
        raise HTTPException(status_code=401, detail="Unauthorized")

    contents = await file.read()
    
    try:
        # Save raw file locally in user-specific folder
        user_dir = os.path.join(settings.DATA_STORE_DIR, "uploads", username)
        os.makedirs(user_dir, exist_ok=True)
        file_path = os.path.join(user_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)
            
        # Determine file type
        ext = file.filename.split('.')[-1].lower()
        if ext == 'csv':
            add_user_file(username, file.filename, "csv", "Ready")
            run_pipeline(file_path)
            message = "Data processed successfully via Azure Pipeline Emulation!"
        elif ext == 'pdf':
            add_user_file(username, file.filename, "pdf", "Ready")
            message = "PDF ingested into vector store!"
        elif ext in ['png', 'jpg', 'jpeg']:
            add_user_file(username, file.filename, "image", "Ready")
            message = "Image saved for Azure Vision processing!"
        else:
            add_user_file(username, file.filename, ext, "Ready")
            message = "File uploaded successfully!"
            
        return {"status": "success", "message": message}
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/files")
async def fetch_files(request: Request):
    """Fetches user uploaded files for Data Hub."""
    try:
        user = get_current_user_from_cookie(request)
        username = user["username"]
    except Exception:
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    files = get_user_files(username)
    return {"files": files}

@router.delete("/files/{filename}")
async def delete_file(filename: str, request: Request):
    """Deletes a user file physically and from DB."""
    try:
        user = get_current_user_from_cookie(request)
        username = user["username"]
    except Exception:
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    file_path = os.path.join(settings.DATA_STORE_DIR, "uploads", username, filename)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        delete_user_file_record(username, filename)
        return {"status": "success", "message": f"{filename} deleted successfully"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/charts/category-trend")
def category_trend():
    path = os.path.join(settings.CURATED_DATA_DIR, "category_timeseries.parquet")
    if not os.path.exists(path):
        return {"labels": [], "categories": [], "datasets": {}}

    df = pd.read_parquet(path)
    df['Month_Year'] = pd.to_datetime(df['Month_Year'])
    df = df.sort_values('Month_Year')

    categories = sorted(df['Category'].dropna().unique().tolist())
    df_pivot = df.pivot_table(
        index='Month_Year', columns='Category', values='Sales'
    ).reset_index()

    datasets = {}
    for cat in categories:
        key = cat.lower().replace(' ', '_')
        datasets[key] = (
            df_pivot[cat].fillna(0).round(0).astype(int).tolist()
            if cat in df_pivot.columns else []
        )

    return {
        "labels":     df_pivot['Month_Year'].dt.strftime('%b-%y').tolist(),
        "categories": categories,
        "datasets":   datasets
    }

@router.get("/charts/segment-region")
def segment_region():
    path = os.path.join(settings.CURATED_DATA_DIR, "segment_region.parquet")
    if not os.path.exists(path):
        return {"regions": [], "segments": [], "datasets": {}}

    df = pd.read_parquet(path)
    regions  = sorted(df['Region'].dropna().unique().tolist())
    segments = sorted(df['Segment'].dropna().unique().tolist())

    datasets = {}
    for seg in segments:
        key = seg.lower().replace(' ', '_')
        datasets[key] = [
            round(
                float(df[
                    (df['Region'] == r) & (df['Segment'] == seg)
                ]['Total_Sales'].sum() / 1000),
                1
            )
            for r in regions
        ]

    return {
        "regions":  regions,
        "segments": segments,
        "datasets": datasets
    }

@router.get("/charts/anomalies")
def anomalies():
    path = os.path.join(settings.CURATED_DATA_DIR, "forecast_anomalies.parquet")
    if not os.path.exists(path):
        return {"points": [], "forecast": []}

    df = pd.read_parquet(path)
    df = df.sort_values('Order Date').reset_index(drop=True)

    return {
        "points": [
            {
                "x":     int(i),
                "y":     round(float(row['Sales']), 2),
                "label": row['Order Date'].strftime('%b-%y'),
                "type":  row['Anomaly_Type'] if pd.notna(row['Anomaly_Type']) else "Normal"
            }
            for i, row in df.iterrows()
        ],
        "forecast": [
            round(float(row['Forecast']), 2) for _, row in df.iterrows()
        ]
    }

@router.get("/charts/basket-rules")
def basket_rules():
    path = os.path.join(settings.CURATED_DATA_DIR, "market_basket_rules.parquet")
    if not os.path.exists(path):
        return {"labels": [], "lift": [], "support": [], "confidence": []}

    df = pd.read_parquet(path)
    df = df.sort_values('lift', ascending=False).head(8)

    return {
        "labels":     (df['antecedents'] + " → " + df['consequents']).tolist(),
        "lift":       df['lift'].round(2).tolist(),
        "support":    df['support'].round(4).tolist() if 'support' in df.columns else [],
        "confidence": df['confidence'].round(4).tolist() if 'confidence' in df.columns else []
    }

@router.get("/charts/kpis")
async def get_kpis():
    forecast_path = os.path.join(settings.CURATED_DATA_DIR, "forecast_anomalies.parquet")
    if not os.path.exists(forecast_path):
        return {"has_data": False}
    
    df = pd.read_parquet(forecast_path)
    
    total_sales = float(df['Sales'].sum())
    
    if 'Anomaly_Flag' in df.columns:
        active_anomalies = int((df['Anomaly_Flag'] == -1).sum())
    elif 'Anomaly_Type' in df.columns:
        anomaly_type = df['Anomaly_Type'].fillna('Normal')
        active_anomalies = int((anomaly_type != 'Normal').sum())
    else:
        active_anomalies = 0
    
    # Forecasted growth: compare last 30 days forecast vs prior 30 days actual
    if 'Order Date' in df.columns:
        df_sorted = df.sort_values('Order Date')
    else:
        df_sorted = df
        
    last_30 = float(df_sorted.tail(30)['Sales'].sum())
    prior_30 = float(df_sorted.iloc[-60:-30]['Sales'].sum()) if len(df_sorted) >= 60 else last_30
    growth = ((last_30 - prior_30) / prior_30 * 100) if prior_30 > 0 else 0.0
    
    return {
        "has_data": True,
        "total_sales": total_sales,
        "active_anomalies": active_anomalies,
        "forecasted_growth": round(growth, 1),
        "currency": "USD"
    }


@router.post("/load_past")
async def load_past_file(username: str, filename: str):
    """Loads a past file and re-triggers pipeline."""
    file_path = os.path.join(settings.DATA_STORE_DIR, f"{username}_{filename}")
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "File not found on server."})
        
    try:
        run_pipeline(file_path)
        return {"message": f"Successfully reloaded {filename} and refreshed analytics."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
