from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RetailAgents:
    def __init__(self, metrics):
        self.metrics = metrics

    def analyst_agent(self):
        return f"Analyst: I have reviewed the data. The model's error rate (RMSE) is {self.metrics.get('rmse')}. Lower is better."

    def forecast_agent(self):
        return "Forecast: Based on the historical trend, I have predicted sales for the next 7 days."

    def strategy_agent(self):
        return "Strategy: Look at the graph. If you see any red dots (anomalies), investigate them immediately. Ensure enough stock for the next 7 days."

    def orchestrate(self):
        return {
            "analysis": self.analyst_agent(),
            "forecast": self.forecast_agent(),
            "strategy": self.strategy_agent()
        }

# A simple brain for the Chatbot to answer questions
documents = [
    "To handle anomalies, check if there was a supply chain delay or a local holiday.",
    "Forecasts are based on historical daily sales using a Linear Regression algorithm.",
    "If sales are dropping, consider running a promotional discount.",
    "The RMSE score tells us how far off our predictions usually are from the actual sales."
]

def rag_chat(query):
    # This finds the closest matching sentence to the user's question
    vectorizer = TfidfVectorizer()
    docs_with_query = documents + [query]
    tfidf_matrix = vectorizer.fit_transform(docs_with_query)
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    best_match_idx = cosine_sim.argmax()
    
    if cosine_sim[best_match_idx] > 0.1: # If it's a decent match
        return documents[best_match_idx]
    return "I am a simple AI. I don't have enough data to answer that specific question."