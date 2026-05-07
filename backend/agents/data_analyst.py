import os
import pandas as pd
from backend.config import settings

def get_data_analyst_response(query: str) -> str:
    """
    Data Analyst Agent.
    In a full production environment, this would use LangChain's create_pandas_dataframe_agent.
    For this implementation, we use a robust heuristic fallback if no LLM is configured, 
    or a simple prompt injection otherwise to ensure safety.
    """
    from backend.agents import get_llm
    
    staged_path = os.path.join(settings.STAGED_DATA_DIR, "train_staged.parquet")
    if not os.path.exists(staged_path):
        return "I don't have access to the data yet. Please upload a dataset."
        
    try:
        # We load a sample to provide schema context to the LLM
        df = pd.read_parquet(staged_path)
        schema_context = f"Columns: {', '.join(df.columns)}. Total rows: {len(df)}."
        
        llm = get_llm()
        if llm:
            prompt = f"""You are a Data Analyst Agent for RetailGPT.
            You have access to a dataset with this schema: {schema_context}
            
            Answer the following user query about the data analytically and professionally.
            If you need to perform calculations that you cannot do, state that you are analyzing the metadata.
            
            Query: {query}
            """
            response = llm.invoke(prompt)
            return response.content
        else:
            return f"[Offline Mode] Data Analyst: I see {len(df)} records. The data covers {df['Category'].nunique()} categories."
    except Exception as e:
        return f"Error analyzing data: {str(e)}"
