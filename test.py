import pytest
from fastapi.testclient import TestClient
from main import app
from core.config import settings

@pytest.fixture(scope="module")
def client():
    
    with TestClient(app) as c:
        yield c

VALID_HEADERS = {"Authorization": f"Bearer {settings.api_key}"}
INVALID_HEADERS = {"Authorization": "Bearer fake-hacker-key"}

def test_unauthorized_access(client):
    response = client.get("/v1/models", headers=INVALID_HEADERS)
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized: Invalid API Key"}

def test_health_check_format(client):
    response = client.get("/health")
    assert response.status_code == 200 
    assert "status" in response.json()

def test_pydantic_context_window_validation(client):
    payload = {
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": settings.n_ctx + 1000 
    }
    response = client.post("/v1/chat/completions", json=payload, headers=VALID_HEADERS)
    
    assert response.status_code == 422
    assert "less_than_equal" in response.text

def test_streaming_done_signal(client):
    payload = {
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 5,
        "stream": True
    }
    
    with client.stream("POST", "/v1/chat/completions", json=payload, headers=VALID_HEADERS) as response:
        assert response.status_code == 200
        chunks = [line for line in response.iter_lines() if line.strip()]
        
        assert chunks[0].startswith("data: {")
        assert chunks[-1] == "data: [DONE]"