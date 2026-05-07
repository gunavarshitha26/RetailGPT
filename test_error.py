import sys
from fastapi.testclient import TestClient
from backend.main import app

try:
    client = TestClient(app)
    response = client.get("/")
    print("Status:", response.status_code)
    print("Response:", response.text)
except Exception as e:
    import traceback
    traceback.print_exc()
