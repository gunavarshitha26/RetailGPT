from backend.agents import get_llm
from backend.agents.data_analyst import get_data_analyst_response
from backend.agents.document_assistant import get_document_assistant_response
from backend.agents.ml_expert import get_ml_expert_response

def route_query(query: str) -> dict:
    """
    LangChain Router.
    Classifies the user's query into one of three domains and delegates to the appropriate agent.
    """
    llm = get_llm()
    
    if not llm:
        # Fallback keyword routing if LLM is not configured
        query_lower = query.lower()
        if "forecast" in query_lower or "model" in query_lower or "anomaly" in query_lower:
            return {"agent": "ML Expert", "response": get_ml_expert_response(query)}
        elif "playbook" in query_lower or "strategy" in query_lower or "handle" in query_lower:
            return {"agent": "Document Assistant", "response": get_document_assistant_response(query)}
        else:
            return {"agent": "Data Analyst", "response": get_data_analyst_response(query)}
            
    # Intelligent LLM-based routing
    system_prompt = """You are a router. Classify the user's query into exactly ONE of the following categories:
    1. 'ml' - if the query is about forecasts, anomalies, models, AI, or predictions.
    2. 'document' - if the query is about strategy, how to handle situations, the playbook, or rules.
    3. 'data' - if the query is about general sales numbers, categories, or raw data analysis.
    
    Respond with ONLY the lowercase category word. Do not include any other text.
    """
    
    try:
        classification = llm.invoke([
            ("system", system_prompt),
            ("human", query)
        ]).content.strip().lower()
        
        if "ml" in classification:
            return {"agent": "ML Expert", "response": get_ml_expert_response(query)}
        elif "document" in classification:
            return {"agent": "Document Assistant", "response": get_document_assistant_response(query)}
        else:
            return {"agent": "Data Analyst", "response": get_data_analyst_response(query)}
            
    except Exception as e:
        return {"agent": "System", "response": f"Routing error: {str(e)}"}
