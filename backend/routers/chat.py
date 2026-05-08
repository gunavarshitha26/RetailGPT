"""RetailGPT AI Copilot chat router with PDF-backed RAG."""
import logging
import os
from dataclasses import dataclass

import pandas as pd
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.dataset_access import load_user_dataset
from backend.config import settings
from backend.routers.auth import get_current_user_from_cookie

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["AI Copilot"])
RAG_PDF_PATH = "RetailGPT_RAG_Knowledge_Base.pdf"


class ChatRequest(BaseModel):
    query: str = ""
    username: str = "user"
    message: str = ""


def get_text_splitter_class():
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter


@dataclass
class TfidfRagStore:
    chunks: list[str]
    vectorizer: object
    matrix: object

    def similarity_search(self, query: str, k: int = 4):
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).ravel()
        top_indices = scores.argsort()[-k:][::-1]
        return [type("Document", (), {"page_content": self.chunks[i]}) for i in top_indices if scores[i] > 0]


def _load_tfidf_rag_store(pdf_path: str):
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from sklearn.feature_extraction.text import TfidfVectorizer

        RecursiveCharacterTextSplitter = get_text_splitter_class()
        pages = PyPDFLoader(pdf_path).load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        ).split_documents(pages)
        texts = [chunk.page_content for chunk in chunks if chunk.page_content.strip()]
        if not texts:
            return None
        vectorizer = TfidfVectorizer(stop_words="english", max_features=6000)
        matrix = vectorizer.fit_transform(texts)
        logger.info("RAG: loaded %s chunks into TF-IDF fallback store.", len(texts))
        return TfidfRagStore(chunks=texts, vectorizer=vectorizer, matrix=matrix)
    except Exception as exc:
        logger.error("RAG TF-IDF fallback load error: %s", exc)
        return None


def load_rag_knowledge_base():
    """Load the RAG PDF into a vector store at startup."""
    if not os.path.exists(RAG_PDF_PATH):
        logger.warning("RAG: %s not found. Proceeding without RAG.", RAG_PDF_PATH)
        return None

    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS

        RecursiveCharacterTextSplitter = get_text_splitter_class()
        loader = PyPDFLoader(RAG_PDF_PATH)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"local_files_only": True},
            )
        except Exception:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        logger.info("RAG: loaded %s chunks into FAISS store.", len(chunks))
        return vectorstore
    except Exception as exc:
        logger.warning("RAG FAISS load failed, trying TF-IDF fallback: %s", exc)
        return _load_tfidf_rag_store(RAG_PDF_PATH)


rag_store = load_rag_knowledge_base()


def get_groq_client():
    if not settings.is_groq_configured:
        return None
    try:
        from groq import Groq

        return Groq(api_key=settings.GROQ_API_KEY)
    except ImportError:
        logger.error("Groq package is not installed.")
        return None


def _money(value) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        amount = 0.0
    return f"${amount:,.2f}"


def _load_chat_dataset(username: str):
    df = load_user_dataset(username, parse_order_date=True)
    if df is None or df.empty:
        return None
    if "Sales" in df.columns:
        df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    return df


def _dataset_context(df: pd.DataFrame) -> str:
    lines = [f"Rows: {len(df):,}", f"Columns: {', '.join(df.columns)}"]
    if "Sales" in df.columns:
        sales = df["Sales"].dropna()
        if not sales.empty:
            lines.extend(
                [
                    f"Total sales revenue: {_money(sales.sum())}",
                    f"Average transaction sales: {_money(sales.mean())}",
                    f"Minimum transaction sales: {_money(sales.min())}",
                    f"Maximum transaction sales: {_money(sales.max())}",
                ]
            )
    if "Order ID" in df.columns:
        lines.append(f"Unique orders: {df['Order ID'].nunique():,}")
    if "Customer ID" in df.columns:
        lines.append(f"Unique customers: {df['Customer ID'].nunique():,}")
    if "Order Date" in df.columns:
        dates = df["Order Date"].dropna()
        if not dates.empty:
            lines.append(f"Date range: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
    for column in ["Category", "Region", "Segment", "Sub-Category"]:
        if column in df.columns:
            values = ", ".join(sorted(str(value) for value in df[column].dropna().unique())[:12])
            lines.append(f"{column} values: {values}")
    return "\n".join(lines)


def _peak_sales_answer(df: pd.DataFrame) -> str | None:
    if "Order Date" not in df.columns or "Sales" not in df.columns:
        return None
    clean = df.dropna(subset=["Order Date", "Sales"]).copy()
    if clean.empty:
        return None
    clean["month"] = clean["Order Date"].dt.strftime("%B")
    clean["quarter"] = "Q" + clean["Order Date"].dt.quarter.astype(str)
    monthly = clean.groupby("month")["Sales"].sum().sort_values(ascending=False)
    quarterly = clean.groupby("quarter")["Sales"].sum().sort_values(ascending=False)
    if monthly.empty:
        return None
    top_month = monthly.index[0]
    top_sales = monthly.iloc[0]
    top_three = ", ".join(f"{month} ({_money(value)})" for month, value in monthly.head(3).items())
    peak_quarter = f"{quarterly.index[0]} ({_money(quarterly.iloc[0])})" if not quarterly.empty else "not available"
    return (
        f"Based on your uploaded dataset, the peak sales month is {top_month} with {_money(top_sales)} in sales.\n\n"
        f"Top months: {top_three}.\n"
        f"Peak quarter/season: {peak_quarter}.\n\n"
        "Recommendation: use this period for inventory planning, promotion timing, and staffing decisions."
    )


def _total_sales_answer(df: pd.DataFrame) -> str | None:
    if "Sales" not in df.columns:
        return None
    sales = df["Sales"].dropna()
    if sales.empty:
        return None
    orders = df["Order ID"].nunique() if "Order ID" in df.columns else "not available"
    rows = f"{len(df):,}"
    return (
        f"Total sales revenue: {_money(sales.sum())}\n"
        f"Average transaction sales: {_money(sales.mean())}\n"
        f"Unique orders: {orders}\n"
        f"Rows analyzed: {rows}"
    )


def _best_region_answer(df: pd.DataFrame) -> str | None:
    if "Region" not in df.columns or "Sales" not in df.columns:
        return None
    region_sales = df.dropna(subset=["Region", "Sales"]).groupby("Region")["Sales"].sum().sort_values(ascending=False)
    if region_sales.empty:
        return None
    total = region_sales.sum()
    best_region = region_sales.index[0]
    best_sales = region_sales.iloc[0]
    share = (best_sales / total * 100) if total else 0
    lines = [f"{region}: {_money(value)}" for region, value in region_sales.head(4).items()]
    return (
        f"Based on your uploaded dataset, the best region is {best_region} with {_money(best_sales)} in sales "
        f"({share:.1f}% of regional sales).\n\n"
        "Region ranking:\n" + "\n".join(lines) + "\n\n"
        "Recommendation: compare the best region's category mix with weaker regions to find expansion opportunities."
    )


def _category_answer(df: pd.DataFrame) -> str | None:
    if "Category" not in df.columns or "Sales" not in df.columns:
        return None
    category_sales = df.dropna(subset=["Category", "Sales"]).groupby("Category")["Sales"].sum().sort_values(ascending=False)
    if category_sales.empty:
        return None
    lines = [f"{category}: {_money(value)}" for category, value in category_sales.head(5).items()]
    return "Top categories in your uploaded dataset:\n\n" + "\n".join(lines)


def _anomaly_answer(df: pd.DataFrame) -> str | None:
    if "Order Date" not in df.columns or "Sales" not in df.columns:
        return None
    try:
        from backend.routers.anomalies import detect_daily_metric_anomalies

        records, timeline = detect_daily_metric_anomalies(df, "Sales")
    except Exception as exc:
        logger.error("Chat anomaly calculation failed: %s", exc)
        return None

    if timeline.empty:
        return None
    if not records:
        return (
            "Based on your uploaded dataset, I did not find strong daily sales anomalies with the current detection method.\n\n"
            "Recommendation: still review Anomaly Center after changing date/category filters, because narrower slices can reveal local anomalies."
        )

    high = sum(1 for record in records if record["severity"] == "high")
    medium = sum(1 for record in records if record["severity"] == "medium")
    top_lines = []
    for record in sorted(records, key=lambda item: abs(item["deviation_pct"]), reverse=True)[:5]:
        top_lines.append(
            f"{record['date']}: actual {_money(record['actual_value'])}, expected {_money(record['expected_value'])}, "
            f"deviation {record['deviation_pct']}%, severity {record['severity']}"
        )
    return (
        f"Based on your uploaded dataset, I found {len(records)} daily sales anomalies. "
        f"High severity: {high}. Medium severity: {medium}.\n\n"
        "Top anomalies:\n" + "\n".join(top_lines) + "\n\n"
        "Recommendation: open Anomaly Center to inspect category, region, and date filters for these spikes or drops."
    )


def _anomaly_reduction_answer(df: pd.DataFrame) -> str:
    context = ""
    if "Order Date" in df.columns and "Sales" in df.columns:
        try:
            from backend.routers.anomalies import detect_daily_metric_anomalies

            records, _ = detect_daily_metric_anomalies(df, "Sales")
            if records:
                top = sorted(records, key=lambda item: abs(item["deviation_pct"]), reverse=True)[0]
                context = (
                    f"\n\nHighest priority in your data: {top['date']} had {_money(top['actual_value'])} actual sales "
                    f"against {_money(top['expected_value'])} expected sales."
                )
        except Exception as exc:
            logger.error("Chat anomaly recommendation calculation failed: %s", exc)

    return (
        "To reduce anomalies, focus on making demand and operations more predictable:\n\n"
        "1. Check stockouts and replenishment delays on anomaly dates.\n"
        "2. Separate planned spikes from real problems, such as promotions, holidays, or bulk orders.\n"
        "3. Set inventory buffers for fast-moving categories before peak months.\n"
        "4. Review low-sales days by region to find local demand drops or fulfillment issues.\n"
        "5. Add alert thresholds so managers investigate spikes/drops immediately.\n"
        "6. Use bundles or discounts for slow-moving items after repeated sales drops."
        f"{context}\n\n"
        "Best next step: open Anomaly Center, filter the top anomaly by date and region, then check whether it was caused by promotion, inventory, or operations."
    )


def _sales_improvement_answer(df: pd.DataFrame) -> str:
    insights = []
    if "Region" in df.columns and "Sales" in df.columns:
        region_sales = df.dropna(subset=["Region", "Sales"]).groupby("Region")["Sales"].sum().sort_values(ascending=False)
        if len(region_sales) >= 2:
            insights.append(f"Scale what works in {region_sales.index[0]} and investigate why {region_sales.index[-1]} is behind.")
    if "Category" in df.columns and "Sales" in df.columns:
        category_sales = df.dropna(subset=["Category", "Sales"]).groupby("Category")["Sales"].sum().sort_values(ascending=False)
        if not category_sales.empty:
            insights.append(f"Prioritize {category_sales.index[0]}, your strongest category, for campaigns and inventory depth.")
    if "Order Date" in df.columns and "Sales" in df.columns:
        peak = _peak_sales_answer(df)
        if peak:
            insights.append("Plan inventory and promotions before the peak month shown by your data.")

    if not insights:
        insights.append("Upload data with Sales, Region, Category, and Order Date columns for more specific recommendations.")

    return (
        "Straight sales improvement actions:\n\n"
        "1. Double down on the best region and copy its product mix into weaker regions.\n"
        "2. Keep high-selling categories in stock before peak season.\n"
        "3. Create bundles from cross-sell pairs to increase basket size.\n"
        "4. Run targeted discounts only for slow-moving categories, not across everything.\n"
        "5. Watch anomaly dates because unexpected drops often point to stock, delivery, or demand issues.\n"
        "6. Use 30-day forecasts to plan inventory instead of reacting late.\n\n"
        "Dataset-specific focus:\n" + "\n".join(f"- {item}" for item in insights[:3])
    )


def _cross_sell_answer(df: pd.DataFrame) -> str | None:
    try:
        from backend.routers.basket import build_rules

        rules, meta = build_rules(df, min_support=0.02, min_confidence=0.3)
        if not rules:
            rules, meta = build_rules(df, min_support=0.01, min_confidence=0.1)
    except Exception as exc:
        logger.error("Chat basket calculation failed: %s", exc)
        rules, meta = [], {}

    if rules:
        lines = []
        for rule in rules[:5]:
            lines.append(
                f"{rule['product_a']} -> {rule['product_b']} "
                f"(lift {rule['lift']:.2f}, confidence {rule['confidence'] * 100:.1f}%, support {rule['support'] * 100:.1f}%)"
            )
        return (
            f"Based on your uploaded dataset, I analyzed {meta.get('total_orders', 0):,} multi-item orders.\n\n"
            "Top cross-sell rules:\n" + "\n".join(lines) + "\n\n"
            "Recommendation: use high-lift pairs for bundles, recommendations, and checkout offers."
        )

    fallback = _cooccurrence_cross_sells(df)
    if fallback:
        return fallback

    return (
        "I could not find reliable cross-sell pairs in your uploaded dataset. The data may not have enough multi-item orders, "
        "or it may be missing Order ID plus Product Name/Sub-Category columns.\n\n"
        "Recommendation: upload order-level transaction data to enable basket analysis."
    )


def _cooccurrence_cross_sells(df: pd.DataFrame) -> str | None:
    if "Order ID" not in df.columns:
        return None
    item_column = "Sub-Category" if "Sub-Category" in df.columns else "Product Name" if "Product Name" in df.columns else None
    if not item_column:
        return None

    tx_df = df[["Order ID", item_column]].dropna().copy()
    tx_df[item_column] = tx_df[item_column].astype(str).str.strip()
    tx_df = tx_df[tx_df[item_column] != ""]
    transactions = tx_df.groupby("Order ID")[item_column].apply(lambda values: sorted(set(values))).tolist()
    transactions = [items for items in transactions if len(items) >= 2]
    total_orders = len(transactions)
    if not total_orders:
        return None

    item_counts = {}
    pair_counts = {}
    for items in transactions:
        for item in items:
            item_counts[item] = item_counts.get(item, 0) + 1
        for index, left in enumerate(items):
            for right in items[index + 1:]:
                pair = (left, right)
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    ranked = []
    for (left, right), pair_count in pair_counts.items():
        support = pair_count / total_orders
        confidence = pair_count / max(item_counts.get(left, 1), 1)
        expected = item_counts.get(right, 0) / total_orders
        lift = confidence / expected if expected else 0
        ranked.append((left, right, pair_count, support, confidence, lift))

    ranked = sorted(ranked, key=lambda row: (row[5], row[2], row[4]), reverse=True)[:5]
    if not ranked:
        return None

    lines = [
        f"{left} -> {right} (lift {lift:.2f}, confidence {confidence * 100:.1f}%, support {support * 100:.1f}%)"
        for left, right, _, support, confidence, lift in ranked
    ]
    return (
        f"Based on your uploaded dataset, I analyzed {total_orders:,} multi-item orders at the {item_column} level.\n\n"
        "Top cross-sell pairs:\n" + "\n".join(lines) + "\n\n"
        "Recommendation: use these pairs for bundles, related-product prompts, and promotional offers."
    )


def _forecast_answer(df: pd.DataFrame) -> str:
    return (
        "Forecasting is available for your uploaded dataset. RetailGPT groups sales by order date and predicts future demand trends.\n\n"
        "Recommendation: open Forecast Studio to choose a horizon such as 7, 30, or 90 days."
    )


def _direct_dataset_answer(query: str, df: pd.DataFrame) -> str | None:
    q = query.lower()
    wants_anomaly = "anomaly" in q or "anomol" in q
    wants_anomaly_reduction = wants_anomaly and any(
        term in q for term in ["reduce", "fix", "handle", "control", "avoid", "prevent", "improve", "what to do", "how to"]
    )
    wants_sales_improvement = any(
        term in q
        for term in [
            "improve sales",
            "increase sales",
            "grow sales",
            "boost sales",
            "improve revenue",
            "increase revenue",
            "recommendation",
            "tips",
        ]
    )
    wants_cross_sell = any(term in q for term in ["cross-sell", "cross sell", "basket", "bought together", "together"])
    wants_peak = any(term in q for term in ["peak", "season", "month"])
    wants_region = "region" in q
    wants_category = "category" in q
    wants_forecast = "forecast" in q
    wants_total_sales = (
        "total sales" in q
        or "total revenue" in q
        or q.strip() in {"sales", "sales?", "revenue", "revenue?", "total sales?", "total sales"}
    )

    answers = []
    multi_question = q.count("?") > 1 or "\n" in query
    if wants_sales_improvement:
        answers.append(("Sales Improvement", _sales_improvement_answer(df)))
    if wants_anomaly_reduction:
        answers.append(("Reduce Anomalies", _anomaly_reduction_answer(df)))
    if wants_total_sales:
        answer = _total_sales_answer(df)
        if answer:
            answers.append(("Total Sales", answer))
    if wants_region:
        answer = _best_region_answer(df)
        if answer:
            answers.append(("Best Region", answer))
    if wants_cross_sell:
        answer = _cross_sell_answer(df)
        if answer:
            answers.append(("Cross-Sells", answer))
    if wants_peak:
        answer = _peak_sales_answer(df)
        if answer:
            answers.append(("Peak Season", answer))
    if wants_anomaly and not wants_anomaly_reduction:
        answer = _anomaly_answer(df)
        if answer:
            answers.append(("Anomalies", answer))
    if wants_category:
        answer = _category_answer(df)
        if answer:
            answers.append(("Categories", answer))
    if wants_forecast:
        answers.append(("Forecast", _forecast_answer(df)))

    if not answers:
        return None
    if len(answers) == 1 and not multi_question:
        return answers[0][1]
    return "\n\n".join(f"{title}\n{answer}" for title, answer in answers)


def _dataset_fallback_response(query: str, df: pd.DataFrame) -> str:
    direct_answer = _direct_dataset_answer(query, df)
    if direct_answer:
        return direct_answer
    return (
        "I found an uploaded dataset, but I need a more specific analytics question to compute the answer. "
        "Try asking about total sales, best region, peak sales month, category performance, forecasts, anomalies, or cross-sells."
    )


def _generic_retail_response(query: str) -> str:
    q = query.lower()
    if "anomaly" in q or "anomol" in q:
        return (
            "In retail, anomalies are unusual events in business metrics, such as a sudden sales spike, a sharp sales drop, "
            "unexpected stockout behavior, or abnormal regional demand.\n\n"
            "What to do: check promotions, inventory, holidays, local events, pricing changes, and supply-chain issues. "
            "Upload a dataset in Data Hub to let RetailGPT detect the exact anomaly dates."
        )
    if "forecast" in q:
        return (
            "Retail forecasting estimates future demand or sales using historical patterns like trend, seasonality, and recent momentum.\n\n"
            "What to do: upload transaction data with Order Date and Sales columns, then use Forecast Studio for 7, 30, or 90 day predictions."
        )
    if "basket" in q or "product" in q or "together" in q:
        return (
            "Market basket analysis finds products that customers often buy together. It helps with bundles, recommendations, store layout, and cross-sell offers.\n\n"
            "What to do: upload order-level retail data with Order ID and Product Name or Sub-Category columns."
        )
    if "sales" in q or "revenue" in q or "total" in q or "peak" in q or "season" in q:
        return (
            "I cannot calculate dataset-specific sales or peak season because no dataset is currently uploaded in Data Hub.\n\n"
            "Generic retail guidance: peak sales seasons often happen around holidays, promotions, payday cycles, and back-to-school or year-end demand. "
            "Upload your CSV to get exact revenue, peak month, category, and region insights."
        )
    return (
        "I can answer general retail analytics questions right now. For dataset-specific answers, upload a CSV in Data Hub.\n\n"
        "You can ask about forecasting, anomaly detection, market basket analysis, inventory planning, promotions, regions, or customer segments."
    )


def _fallback_response(query: str, df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return _generic_retail_response(query)
    return _dataset_fallback_response(query, df)


async def _chat_impl(request: ChatRequest, user=Depends(get_current_user_from_cookie)):
    user_query = (request.message or request.query or "").strip()
    if not user_query:
        return {"response": "Please send a question about your retail dataset."}

    df = _load_chat_dataset(user["username"])
    if df is None:
        return {"response": _generic_retail_response(user_query)}
    direct_answer = _direct_dataset_answer(user_query, df)
    if direct_answer:
        return {"response": direct_answer}

    context = ""
    if df is not None and rag_store:
        try:
            docs = rag_store.similarity_search(user_query, k=4)
            context = "\n\n".join([doc.page_content for doc in docs])
        except Exception as exc:
            logger.error("RAG retrieval error: %s", exc)

    dataset_context = _dataset_context(df) if df is not None else "No user dataset is currently uploaded."
    system_prompt = f"""You are the RetailGPT AI Copilot - an expert retail analytics assistant.

CURRENT USER DATASET CONTEXT:
{dataset_context}

REFERENCE KNOWLEDGE CONTEXT (use only for generic retail explanations, not for user-specific numbers):
{context}

RULES:
- If no user dataset is uploaded, do not provide dataset-specific totals, customer counts, dates,
  product rankings, or Superstore knowledge-base numbers. Give generic retail guidance and ask the
  user to upload a CSV for exact analysis.
- If a user dataset is uploaded, answer dataset-specific questions using CURRENT USER DATASET CONTEXT.
- Use REFERENCE KNOWLEDGE CONTEXT only to explain retail concepts and methods.
- For anomaly questions, explain anomaly detection as unusual spikes/drops and mention RetailGPT's
  Anomaly Center.
- Keep responses concise and actionable. Use bullet points for lists.
- Always end with a brief actionable recommendation when relevant.
"""

    groq_client = get_groq_client()
    if not groq_client:
        return {"response": _fallback_response(user_query, df)}

    try:
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            max_tokens=500,
            temperature=0.3,
        )
        return {"response": response.choices[0].message.content}
    except Exception as exc:
        logger.error("Groq API error: %s", exc)
        return {"response": _fallback_response(user_query, df)}


@router.post("")
async def chat_no_slash(request: ChatRequest, user=Depends(get_current_user_from_cookie)):
    return await _chat_impl(request, user)


@router.post("/")
async def chat_with_slash(request: ChatRequest, user=Depends(get_current_user_from_cookie)):
    return await _chat_impl(request, user)
