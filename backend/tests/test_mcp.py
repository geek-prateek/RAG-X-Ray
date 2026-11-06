"""Tests for RAG X-Ray Inspector & Evaluator MCP server endpoints and tools."""
from fastapi.testclient import TestClient
from app.main import app

def test_mcp_sse_endpoint():
    with TestClient(app) as client:
        with client.stream("GET", "/mcp/sse") as response:
            assert response.status_code == 200
            first_line = next(response.iter_lines())
            assert "event" in first_line or "endpoint" in first_line

def test_mcp_compare_tool():
    from app.mcp_server import compare_semantic_similarity
    import json

    res = compare_semantic_similarity("Annual leave policy", "How many days off can I take?")
    data = json.loads(res)
    assert "cosine_similarity" in data
    assert "diagnostic_verdict" in data
    assert data["dimensions"] == 384

def test_mcp_inspect_vector_tool():
    from app.mcp_server import inspect_vector_embedding
    import json

    res = inspect_vector_embedding("Quick brown fox jumps")
    data = json.loads(res)
    assert data["vector_dimensions"] == 384
    assert len(data["first_8_dimensions_preview"]) == 8
