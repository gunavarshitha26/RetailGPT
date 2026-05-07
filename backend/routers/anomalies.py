import math
import os
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query

from backend.config import settings
from backend.database import get_user_files
from backend.routers.auth import get_current_user_from_cookie

router = APIRouter(tags=["Anomaly Center"])

ALLOWED_METRICS = {"Sales"}
ALLOWED_SEVERITIES = {"all", "high", "medium", "low"}


def get_user_dataset_path(username: str) -> Optional[str]:
    for file_record in get_user_files(username):
        if file_record.get("file_type") != "csv" or file_record.get("status") != "Ready":
            continue
        path = os.path.join(settings.DATA_STORE_DIR, "uploads", username, file_record["filename"])
        if os.path.exists(path):
            return path
    return None


def load_user_dataset(username: str) -> Optional[pd.DataFrame]:
    path = get_user_dataset_path(username)
    if not path:
        return None

    df = pd.read_csv(path)
    if "Order Date" not in df.columns:
        return None

    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Order Date"])
    for column in ["Sales"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def option_values(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    return sorted(str(value) for value in df[column].dropna().unique().tolist())


def severity_from_deviation(deviation_pct: float) -> str:
    deviation = abs(float(deviation_pct))
    if deviation >= 80:
        return "high"
    if deviation >= 40:
        return "medium"
    return "low"


def finite_float(value, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def representative_dimension(df: pd.DataFrame, date_value: pd.Timestamp, metric: str) -> tuple[str, str]:
    day_rows = df[df["Order Date"].dt.date == date_value.date()].copy()
    if day_rows.empty:
        return "All", "All"
    if metric in day_rows.columns:
        day_rows["_rank_metric"] = day_rows[metric].abs().fillna(0)
        row = day_rows.sort_values("_rank_metric", ascending=False).iloc[0]
    else:
        row = day_rows.iloc[0]
    return str(row.get("Category", "All") or "All"), str(row.get("Region", "All") or "All")


def anomaly_mask(values: pd.Series) -> pd.Series:
    cleaned = values.fillna(0).astype(float)
    if len(cleaned) < 8 or cleaned.std(ddof=0) == 0:
        z_scores = pd.Series(np.zeros(len(cleaned)), index=cleaned.index)
        return z_scores.abs() > 2.5

    try:
        from sklearn.ensemble import IsolationForest

        model = IsolationForest(contamination=0.05, random_state=42)
        predictions = model.fit_predict(cleaned.to_numpy().reshape(-1, 1))
        return pd.Series(predictions == -1, index=cleaned.index)
    except Exception:
        z_scores = (cleaned - cleaned.mean()) / cleaned.std(ddof=0)
        return z_scores.abs() > 2.5


def expected_series(values: pd.Series) -> pd.Series:
    rolling = values.rolling(window=14, center=True, min_periods=3).median()
    return rolling.fillna(values.median()).fillna(0)


def detect_daily_metric_anomalies(df: pd.DataFrame, metric: str) -> tuple[list[dict], pd.DataFrame]:
    if metric not in df.columns:
        return [], pd.DataFrame(columns=["date", "value", "expected", "lower", "upper", "is_anomaly"])

    daily = (
        df.groupby(df["Order Date"].dt.date)[metric]
        .sum()
        .reset_index(name="value")
    )
    daily["date"] = pd.to_datetime(daily["Order Date"])
    daily = daily.drop(columns=["Order Date"]).sort_values("date").reset_index(drop=True)
    daily["expected"] = expected_series(daily["value"])
    residuals = daily["value"] - daily["expected"]
    residual_std = finite_float(residuals.std(ddof=0))
    band = max(residual_std * 2.5, finite_float(daily["value"].std(ddof=0)) * 0.25)
    daily["lower"] = daily["expected"] - band
    daily["upper"] = daily["expected"] + band
    daily["is_anomaly"] = anomaly_mask(daily["value"]) | (residuals.abs() > band)

    records = []
    for _, row in daily[daily["is_anomaly"]].iterrows():
        expected = finite_float(row["expected"])
        actual = finite_float(row["value"])
        denominator = abs(expected) if abs(expected) > 1e-9 else 1.0
        deviation_pct = ((actual - expected) / denominator) * 100
        category, region = representative_dimension(df, row["date"], metric)
        records.append(
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "category": category,
                "region": region,
                "metric": metric,
                "actual_value": round(actual, 2),
                "expected_value": round(expected, 2),
                "expected_lower": round(finite_float(row["lower"]), 2),
                "expected_upper": round(finite_float(row["upper"]), 2),
                "deviation_pct": round(deviation_pct, 1),
                "severity": severity_from_deviation(deviation_pct),
            }
        )
    return records, daily


def apply_filters(records: list[dict], severity: str, category: str, region: str) -> list[dict]:
    filtered = records
    if severity in ALLOWED_SEVERITIES and severity != "all":
        filtered = [record for record in filtered if record["severity"] == severity]
    if category != "All":
        filtered = [record for record in filtered if record["category"] == category]
    if region != "All":
        filtered = [record for record in filtered if record["region"] == region]
    return sorted(filtered, key=lambda record: abs(record["deviation_pct"]), reverse=True)


@router.get("/api/anomalies")
async def get_anomalies(
    metric: str = Query("Sales"),
    severity: str = Query("all"),
    category: str = Query("All"),
    region: str = Query("All"),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    user=Depends(get_current_user_from_cookie),
):
    df = load_user_dataset(user["username"])
    if df is None or df.empty:
        return {"has_data": False, "anomalies": [], "timeline": []}

    metric = metric if metric in ALLOWED_METRICS else "Sales"
    severity = severity.lower() if severity.lower() in ALLOWED_SEVERITIES else "all"
    categories = option_values(df, "Category")
    regions = option_values(df, "Region")
    all_dates_before_filter = pd.to_datetime(df["Order Date"], errors="coerce").dropna()

    if metric not in df.columns:
        return {
            "has_data": True,
            "metric_available": False,
            "message": f"{metric} is not available in the uploaded dataset.",
            "metric": metric,
            "summary": {
                "total": 0,
                "high": 0,
                "medium": 0,
                "date_range": (
                    f"{all_dates_before_filter.min().strftime('%Y-%m-%d')} to {all_dates_before_filter.max().strftime('%Y-%m-%d')}"
                    if not all_dates_before_filter.empty
                    else "No dates"
                ),
            },
            "anomalies": [],
            "timeline": [],
            "filters": {
                "categories": categories,
                "regions": regions,
                "min_date": all_dates_before_filter.min().strftime("%Y-%m-%d") if not all_dates_before_filter.empty else "",
                "max_date": all_dates_before_filter.max().strftime("%Y-%m-%d") if not all_dates_before_filter.empty else "",
            },
        }

    if start_date:
        df = df[df["Order Date"] >= pd.to_datetime(start_date, errors="coerce")]
    if end_date:
        df = df[df["Order Date"] <= pd.to_datetime(end_date, errors="coerce")]

    if category != "All" and "Category" in df.columns:
        df = df[df["Category"] == category]
    if region != "All" and "Region" in df.columns:
        df = df[df["Region"] == region]

    metric_records, timeline_df = detect_daily_metric_anomalies(df, "Sales" if metric == "Sales" else metric)
    records = metric_records

    records = apply_filters(records, severity, category, region)
    timeline = [
        {
            "date": row["date"].strftime("%Y-%m-%d"),
            "value": round(finite_float(row["value"]), 2),
            "expected": round(finite_float(row["expected"]), 2),
            "lower": round(finite_float(row["lower"]), 2),
            "upper": round(finite_float(row["upper"]), 2),
            "is_anomaly": bool(row["is_anomaly"]),
            "deviation_pct": round(
                ((finite_float(row["value"]) - finite_float(row["expected"])) / max(abs(finite_float(row["expected"])), 1.0)) * 100,
                1,
            ),
        }
        for _, row in timeline_df.iterrows()
    ]

    all_dates = pd.to_datetime(df["Order Date"], errors="coerce").dropna()
    summary = {
        "total": len(records),
        "high": sum(1 for record in records if record["severity"] == "high"),
        "medium": sum(1 for record in records if record["severity"] == "medium"),
        "date_range": (
            f"{all_dates.min().strftime('%Y-%m-%d')} to {all_dates.max().strftime('%Y-%m-%d')}"
            if not all_dates.empty
            else "No dates"
        ),
    }

    return {
        "has_data": True,
        "metric": metric,
        "summary": summary,
        "anomalies": records,
        "timeline": timeline,
        "filters": {
            "categories": categories,
            "regions": regions,
            "min_date": all_dates.min().strftime("%Y-%m-%d") if not all_dates.empty else "",
            "max_date": all_dates.max().strftime("%Y-%m-%d") if not all_dates.empty else "",
        },
    }
