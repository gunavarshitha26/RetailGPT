import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from backend.config import settings

def _init_chroma():
    """Initializes ChromaDB with the Playbook if it doesn't exist."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Check if we already have a persisted vector store
    if os.path.exists(settings.CHROMA_PERSIST_DIR) and os.listdir(settings.CHROMA_PERSIST_DIR):
        return Chroma(persist_directory=settings.CHROMA_PERSIST_DIR, embedding_function=embeddings)
        
    # Otherwise, load and embed the PDF
    if not os.path.exists(settings.PLAYBOOK_PATH):
        print(f"Warning: Playbook not found at {settings.PLAYBOOK_PATH}")
        return None
        
    print("Ingesting Superstore Playbook into ChromaDB...")
    loader = PyPDFLoader(settings.PLAYBOOK_PATH)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR
    )
    vectorstore.persist()
    return vectorstore

def get_document_assistant_response(query: str) -> str:
    """
    Document Assistant Agent (RAG).
    Searches ChromaDB for Playbook context and answers the query.
    """
    from backend.agents import get_llm
    
    vectorstore = _init_chroma()
    if not vectorstore:
        return "I cannot access the Superstore Playbook right now."
        
    # Retrieve relevant chunks
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])
    
    llm = get_llm()
    if llm:
        prompt = f"""You are a Strategic Retail Assistant. 
        Use the following excerpts from the Superstore Operations Playbook to answer the user's question.
        If the answer is not in the context, say "I don't have information on that in the playbook."
        
        Context:
        {context}
        
        Question: {query}
        """
        response = llm.invoke(prompt)
        return response.content
    else:
        return f"[Offline Mode] Document Assistant found relevant info:\n{context[:300]}..."
