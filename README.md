# RetailGPT: Enterprise Decision Intelligence Platform

RetailGPT is an advanced, AI-powered multi-agent retail analytics platform. It features an Azure-emulated data engineering pipeline, a LangChain-based orchestrator, an interactive natively-served web frontend, and a secure modular FastAPI backend.

> [!NOTE]
> For a deep technical dive, refer to the [Project Overview PDF](file:///c:/Users/gunav/Downloads/Primary%20Domain/RetailGPT/RetailGPT_Project_Overview.pdf) generated in the root directory.

## 🚀 Prerequisites

1. **Python 3.11+** installed on your machine.
2. (Optional but recommended) A free [Groq API Key](https://console.groq.com/) for powering the advanced multi-agent LangChain chatbot. Without this, the chatbot will run in a safe "offline fallback" mode.

---

## 🛠️ Setup Instructions

### 1. Environment Setup

Clone the repository and install the dependencies:

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configuration

Copy the example environment file:
```bash
# On Windows:
copy .env.example .env

# On Mac/Linux:
cp .env.example .env
```
Open `.env` in a text editor and paste your Groq API key into the `GROQ_API_KEY` variable.

---

## ▶️ Running the Application Locally

The application now serves both the backend API and the frontend UI from a single unified FastAPI instance. There is no longer a separate Streamlit frontend!

### Start the Server
From the root directory of the project, run:

```bash
# 1. Make sure your virtual environment is activated
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# 2. Start the unified server
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Web Platform:** Access the sleek UI natively at [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **API Documentation:** View the interactive endpoints at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

*(Note: If you encounter a `ModuleNotFoundError` during startup, simply run `pip install -r requirements.txt` again to ensure you have the latest dependencies like `jinja2` and `python-jose`.)*

---

## 🐋 Running via Docker (Alternative)

If you prefer to run everything in a container:

```bash
# Build the image
docker build -t retailgpt .

# Run the container
docker run -p 8000:8000 retailgpt
```
Access the UI at `http://localhost:8000`.

---

## 📖 How to Use the Platform

1. **Sign Up**: Upon launching the UI (`http://127.0.0.1:8000/`), select the "Sign Up" tab and create a new corporate account.
2. **Log In**: Switch to the "Log In" tab and authenticate with your new credentials.
3. **View Dashboard**: You will be redirected to the Executive Dashboard.
   - **Intelligent Visibility**: If no data is uploaded, the dashboard will guide you with a full-page empty state.
   - **Dynamic KPIs**: View live Total Sales, Active Anomalies, and Growth metrics fetched directly from the backend.
   - **Enhanced Visualization**: Explore the clean, line-based Sales Forecast & Anomalies chart.
   - **Market Basket Analysis**: Check association rules with improved label formatting and dynamic scaling.
4. **Data Management**: Use the sidebar to initiate dataset uploads. The platform automatically processes your files through the ML pipeline.
5. **Ask the Copilot**: Use the floating chat widget in the bottom right corner to interact with the multi-agent AI system.
   - Example 1: *"What does the ML model forecast?"* (Routes to ML Expert)
   - Example 2: *"How should I handle a sales spike in Technology?"* (Routes to Playbook RAG Assistant)
   - Example 3: *"What is in the data?"* (Routes to Data Analyst)

---

## 📊 Exporting for Power BI

To export the curated, machine-learning-enriched data into a format optimized for Power BI:

```bash
python -m pipeline.export_powerbi
```
This will generate normalized CSV files in the `powerbi_export/` directory that you can import directly into your BI dashboards.
