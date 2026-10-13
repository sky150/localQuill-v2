import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.api.deps import embeddings
from src.api.main import app
from src.api.routers import chat as chat_router
from src.api.routers import meta as meta_router
from src.vector_db import documents

FAKE_EMBEDDINGS = DeterministicFakeEmbedding(size=16)


@pytest.fixture
def client(tmp_path, monkeypatch):
    chroma_path = str(tmp_path / "chroma")
    monkeypatch.setattr(documents, "UPLOAD_PATH", str(tmp_path / "uploads"))
    monkeypatch.setattr(documents, "CHROMA_PATH", chroma_path)
    monkeypatch.setattr(meta_router, "CHROMA_PATH", chroma_path)
    monkeypatch.setattr(meta_router, "OLLAMA_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    app.dependency_overrides[embeddings] = lambda: FAKE_EMBEDDINGS
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def upload(
    client,
    name="notes.md",
    content=b"# Notes\n\nSome text to index.",
    kind="userdocs",
):
    return client.post(
        f"/api/documents?kind={kind}",
        files={"file": (name, content, "text/markdown")},
    )


def test_upload_list_delete_roundtrip(client):
    response = upload(client)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["added"] > 0
    assert body["total"] == body["added"]

    listing = client.get("/api/documents?kind=userdocs").json()
    assert len(listing) == 1
    assert listing[0]["filename"] == "notes.md"
    assert listing[0]["chunks"] == body["total"]

    again = upload(client)
    assert again.status_code == 200, again.text
    assert again.json()["total"] == body["total"]

    deleted = client.delete(f"/api/documents/{body['doc_id']}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted_chunks"] == body["total"]
    assert client.get("/api/documents?kind=userdocs").json() == []


def test_upload_rejects_unsupported_and_unknown_kind(client):
    assert upload(client, name="bad.xyz").status_code == 400
    assert upload(client, kind="nope").status_code == 400


def test_delete_missing_document(client):
    assert client.delete("/api/documents/deadbeef").status_code == 404


def test_chat_validation(client):
    assert (
        client.post("/api/chat", json={"task": "nope", "text": "hi"}).status_code == 400
    )
    assert (
        client.post(
            "/api/chat", json={"task": "qa", "style": "nope", "text": "hi"}
        ).status_code
        == 400
    )
    assert client.post("/api/chat", json={"task": "qa", "text": "  "}).status_code == 422


def test_chat_success(client, monkeypatch):
    class FakeTask:
        def run(self, text, **opts):
            return f"ok:{text}"

    monkeypatch.setattr(chat_router, "get_task", lambda name, style="formal": FakeTask())
    response = client.post(
        "/api/chat", json={"task": "qa", "style": "formal", "text": "hello"}
    )
    assert response.status_code == 200, response.text
    assert response.json()["result"] == "ok:hello"


def test_health_reports_collections(client):
    upload(client)
    body = client.get("/api/health").json()
    assert body["collections"]["userdocs"] > 0
    assert body["ollama"] is False
