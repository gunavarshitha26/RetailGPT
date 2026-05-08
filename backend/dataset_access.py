import os
from typing import Optional

import pandas as pd

from backend.config import settings
from backend.database import get_user_files


def get_user_dataset_path(username: str) -> Optional[str]:
    for file_record in get_user_files(username):
        if file_record.get("file_type") != "csv" or file_record.get("status") != "Ready":
            continue
        filename = file_record.get("filename")
        if not filename:
            continue
        path = os.path.join(settings.DATA_STORE_DIR, "uploads", username, filename)
        if os.path.exists(path):
            return path
    return None


def has_user_dataset(username: str) -> bool:
    return bool(get_user_dataset_path(username))


def load_user_dataset(username: str, parse_order_date: bool = False) -> Optional[pd.DataFrame]:
    path = get_user_dataset_path(username)
    if not path:
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if parse_order_date:
        if "Order Date" not in df.columns:
            return None
        df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Order Date"])

    return df
