"""RetailGPT AI Copilot chat router with PDF-backed RAG."""
import logging
import os
from dataclasses import dataclass

from fastapi import APIRouter, Depends
from pydantic import BaseModel

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


def _fallback_response(query: str, context: str) -> str:
    if context:
        snippet = context[:700].strip()
        return (
            "Based on the current RetailGPT knowledge base:\n\n"
            f"{snippet}\n\n"
            "Recommendation: configure GROQ_API_KEY to enable full synthesized Copilot answers."
        )

    q = query.lower()
    if "sales" in q or "revenue" in q:
        return "This information isn't in the current dataset. Please upload updated data in the Data Hub."
    if "forecast" in q:
        return "LightGBM forecasting is available after a curated dataset is loaded. Recommendation: upload a CSV in Data Hub, then review the Dashboard forecast."
    if "anomaly" in q:
        return "Anomaly detection uses Isolation Forest with IQR-style business interpretation. Recommendation: upload current sales data to surface spikes and drops."
    return "This information isn't in the current dataset. Please upload updated data in the Data Hub."


async def _chat_impl(request: ChatRequest, user=Depends(get_current_user_from_cookie)):
    user_query = (request.message or request.query or "").strip()
    if not user_query:
        return {"response": "Please send a question about your retail dataset."}

    context = ""
    if rag_store:
        try:
            docs = rag_store.similarity_search(user_query, k=4)
            context = "\n\n".join([doc.page_content for doc in docs])
        except Exception as exc:
            logger.error("RAG retrieval error: %s", exc)

    system_prompt = f"""You are the RetailGPT AI Copilot - an expert retail analytics assistant
for a US Superstore enterprise platform. You have access to deep knowledge about the dataset.

KNOWLEDGE BASE CONTEXT (use this to answer accurately):
{context}

RULES:
- Answer questions about sales, forecasts, anomalies, products, regions, and segments using
  the knowledge base data above.
- Give specific numbers and percentages when available.
- For anomaly questions, explain the Isolation Forest + IQR method.
- For product questions, reference the top products and sub-categories from the knowledge base.
- For seasonal questions, reference the Q4 peak pattern and monthly trends.
- If the question is outside the knowledge base, say: "This information isn't in the current
  dataset. Please upload updated data in the Data Hub."
- Keep responses concise and actionable. Use bullet points for lists.
- Always end with a brief actionable recommendation when relevant.
"""

    groq_client = get_groq_client()
    if not groq_client:
        return {"response": _fallback_response(user_query, context)}

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
        return {"response": _fallback_response(user_query, context)}


@router.post("")
async def chat_no_slash(request: ChatRequest, user=Depends(get_current_user_from_cookie)):
    return await _chat_impl(request, user)


@router.post("/")
async def chat_with_slash(request: ChatRequest, user=Depends(get_current_user_from_cookie)):
    return await _chat_impl(request, user)
