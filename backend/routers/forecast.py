from datetime import timedelta

import pandas as pd
from fastapi import APIRouter, Depends, Query

from backend.dataset_access import get_user_dataset_path, load_user_dataset
from backend.routers.auth import get_current_user_from_cookie

router = APIRouter(tags=["Forecast Studio"])

ALLOWED_METRICS = {"Sales"}
ALLOWED_HORIZONS = {7, 30, 90}
ALLOWED_CATEGORIES = {"All", "Furniture", "Office Supplies", "Technology"}
ALLOWED_REGIONS = {"All", "East", "West", "Central", "South"}


def get_forecast_source_path(username: str):
    return get_user_dataset_path(username)


def load_forecast_source(username: str):
    return load_user_dataset(username, parse_order_date=True)


def make_linear_forecast(daily_df, metric, horizon):
    from sklearn.linear_model import LinearRegression

    model_df = daily_df.copy().reset_index(drop=True)
    model_df["ordinal"] = model_df["date"].map(pd.Timestamp.toordinal)
    x = model_df[["ordinal"]]
    y = model_df[metric]

    model = LinearRegression()
    model.fit(x, y)
    residuals = y - model.predict(x)
    residual_std = float(residuals.std()) if len(residuals) > 1 else 0.0

    last_date = model_df["date"].max()
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=horizon, freq="D")
    future_ordinals = pd.DataFrame({"ordinal": future_dates.map(pd.Timestamp.toordinal)})
    preds = model.predict(future_ordinals)

    return [
        {
            "date": date.strftime("%Y-%m-%d"),
            "value": round(max(float(pred), 0.0), 2),
            "lower": round(max(float(pred - 1.96 * residual_std), 0.0), 2),
            "upper": round(max(float(pred + 1.96 * residual_std), 0.0), 2),
        }
        for date, pred in zip(future_dates, preds)
    ]


def make_prophet_forecast(daily_df, metric, horizon):
    try:
        from prophet import Prophet
    except ImportError:
        return None

    prophet_df = daily_df.rename(columns={"date": "ds", metric: "y"})
    model = Prophet(interval_width=0.8, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=horizon, freq="D")
    forecast = model.predict(future).tail(horizon)
    return [
        {
            "date": row["ds"].strftime("%Y-%m-%d"),
            "value": round(max(float(row["yhat"]), 0.0), 2),
            "lower": round(max(float(row["yhat_lower"]), 0.0), 2),
            "upper": round(max(float(row["yhat_upper"]), 0.0), 2),
        }
        for _, row in forecast.iterrows()
    ]


@router.get("/api/forecast")
async def get_forecast(
    metric: str = Query("Sales"),
    horizon: int = Query(30),
    category: str = Query("All"),
    region: str = Query("All"),
    user=Depends(get_current_user_from_cookie),
):
    df = load_forecast_source(user["username"])
    if df is None or df.empty:
        return {"has_data": False, "historical": [], "forecast": []}

    metric = metric if metric in ALLOWED_METRICS else "Sales"
    horizon = horizon if horizon in ALLOWED_HORIZONS else 30
    category = category if category in ALLOWED_CATEGORIES else "All"
    region = region if region in ALLOWED_REGIONS else "All"

    if category != "All" and "Category" in df.columns:
        df = df[df["Category"] == category]
    if region != "All" and "Region" in df.columns:
        df = df[df["Region"] == region]

    if df.empty:
        return {"has_data": True, "metric_available": True, "historical": [], "forecast": []}

    if metric not in df.columns:
        return {
            "has_data": True,
            "metric_available": False,
            "message": f"{metric} is not available in the uploaded dataset.",
            "historical": [],
            "forecast": [],
        }

    daily = (
        df.groupby(df["Order Date"].dt.date)[metric]
        .sum()
        .reset_index(name=metric)
        .rename(columns={"Order Date": "date"})
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")

    historical = [
        {"date": row["date"].strftime("%Y-%m-%d"), "value": round(float(row[metric]), 2)}
        for _, row in daily.iterrows()
    ]
    forecast = make_prophet_forecast(daily, metric, horizon) or make_linear_forecast(daily, metric, horizon)

    return {
        "has_data": True,
        "metric_available": True,
        "metric": metric,
        "horizon": horizon,
        "category": category,
        "region": region,
        "historical": historical,
        "forecast": forecast,
    }
