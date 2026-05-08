import math
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, Query

from backend.dataset_access import get_user_dataset_path as find_user_dataset_path
from backend.dataset_access import load_user_dataset as read_user_dataset
from backend.routers.auth import get_current_user_from_cookie

router = APIRouter(tags=["Basket Intelligence"])


def get_user_dataset_path(username: str) -> Optional[str]:
    return find_user_dataset_path(username)


def load_user_dataset(username: str) -> Optional[pd.DataFrame]:
    return read_user_dataset(username)


def option_values(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    return sorted(str(value) for value in df[column].dropna().unique().tolist())


def finite_float(value, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def format_itemset(itemset) -> list[str]:
    return sorted(str(item) for item in itemset)


def build_rules(df: pd.DataFrame, min_support: float, min_confidence: float) -> tuple[list[dict], dict]:
    required = {"Order ID"}
    if not required.issubset(df.columns):
        return [], {"total_orders": 0, "avg_basket_size": 0.0}

    item_column = "Product Name" if "Product Name" in df.columns else "Sub-Category" if "Sub-Category" in df.columns else None
    if not item_column:
        return [], {"total_orders": 0, "avg_basket_size": 0.0}

    tx_df = df[["Order ID", item_column]].dropna().copy()
    tx_df[item_column] = tx_df[item_column].astype(str).str.strip()
    tx_df = tx_df[tx_df[item_column] != ""]
    transactions = (
        tx_df.groupby("Order ID")[item_column]
        .apply(lambda values: sorted(set(values)))
        .tolist()
    )
    transactions = [items for items in transactions if len(items) >= 2]
    total_orders = len(transactions)
    avg_basket_size = sum(len(items) for items in transactions) / total_orders if total_orders else 0.0
    meta = {"total_orders": total_orders, "avg_basket_size": avg_basket_size}
    if not transactions:
        return [], meta

    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder

        encoder = TransactionEncoder()
        matrix = encoder.fit(transactions).transform(transactions)
        basket_df = pd.DataFrame(matrix, columns=encoder.columns_)
        frequent = apriori(basket_df, min_support=min_support, use_colnames=True)
        if frequent.empty:
            return [], meta
        rules_df = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
    except Exception:
        return [], meta

    if rules_df.empty:
        return [], meta

    rules_df = rules_df[
        (rules_df["antecedents"].apply(len) >= 1) &
        (rules_df["consequents"].apply(len) >= 1)
    ].copy()
    rules_df = rules_df.sort_values(["lift", "confidence", "support"], ascending=False)

    records = []
    for _, row in rules_df.iterrows():
        antecedents = format_itemset(row["antecedents"])
        consequents = format_itemset(row["consequents"])
        records.append(
            {
                "antecedents": antecedents,
                "consequents": consequents,
                "product_a": ", ".join(antecedents),
                "product_b": ", ".join(consequents),
                "support": round(finite_float(row["support"]), 5),
                "confidence": round(finite_float(row["confidence"]), 5),
                "lift": round(finite_float(row["lift"]), 3),
            }
        )
    return records, meta


@router.get("/api/basket")
async def get_basket(
    min_support: float = Query(0.02, ge=0.01, le=0.1),
    min_confidence: float = Query(0.3, ge=0.1, le=0.9),
    category: str = Query("All"),
    max_rules: int = Query(25),
    user=Depends(get_current_user_from_cookie),
):
    df = load_user_dataset(user["username"])
    if df is None or df.empty:
        return {"has_data": False, "summary": {}, "rules": []}

    categories = option_values(df, "Category")
    if category != "All" and "Category" in df.columns:
        df = df[df["Category"].astype(str) == category]

    rules, meta = build_rules(df, min_support, min_confidence)
    limited_rules = rules[: max(1, min(max_rules, 50))]
    strongest = limited_rules[0] if limited_rules else None

    return {
        "has_data": True,
        "summary": {
            "total_orders": meta["total_orders"],
            "total_rules": len(rules),
            "strongest_rule": (
                f"{strongest['product_a']} -> {strongest['product_b']} ({strongest['lift']:.2f} lift)"
                if strongest
                else "No rule found"
            ),
            "avg_basket_size": round(meta["avg_basket_size"], 2),
        },
        "rules": limited_rules,
        "filters": {"categories": categories},
    }
