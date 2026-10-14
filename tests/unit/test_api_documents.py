import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.api.deps import embeddings
from src.api.main import app
from src.api.routers import chat as chat_router
from src.api.routers import meta as meta_router
from src.vector_db import documents
from src.workspace import files as workspace_files

FAKE_EMBEDDINGS = DeterministicFakeEmbedding(size=16)


@pytest.fixture
def client(tmp_path, monkeypatch):
    chroma_path = str(tmp_path / "chroma")
    monkeypatch.setattr(documents, "UPLOAD_PATH", str(tmp_path / "uploads"))
    monkeypatch.setattr(documents, "CHROMA_PATH", chroma_path)
    monkeypatch.setattr(meta_router, "CHROMA_PATH", chroma_path)
    monkeypatch.setattr(meta_router, "OLLAMA_URL", "http://127.0.0.1:1")
    monkeypatch.setattr(workspace_files, "WORKSPACE_PATH", str(tmp_path / "workspace"))
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


def test_set_outline_from_workspace(client):
    client.post("/api/files", json={"name": "Outline.md", "kind": "file"})
    client.put(
        "/api/files/content",
        json={"path": "Outline.md", "content": "1. Intro\n2. Body"},
    )

    response = client.post(
        "/api/documents/from-workspace", json={"path": "Outline.md", "kind": "outline"}
    )
    assert response.status_code == 200, response.text

    outline = client.get("/api/documents?kind=outline").json()
    assert len(outline) == 1
    assert outline[0]["filename"] == "Outline.md"

    deleted = client.delete(f"/api/documents/{outline[0]['doc_id']}")
    assert deleted.status_code == 200
    still_there = client.get("/api/files/content", params={"path": "Outline.md"})
    assert still_there.status_code == 200


def test_chat_passes_attached_context(client, monkeypatch):
    client.post("/api/files", json={"name": "alice.md", "kind": "file"})
    client.put(
        "/api/files/content",
        json={"path": "alice.md", "content": "Alice was released in 2010"},
    )

    captured = {}

    class FakeTask:
        def run(self, text, **opts):
            captured.update(opts)
            return "ok"

    monkeypatch.setattr(chat_router, "get_task", lambda *args, **kwargs: FakeTask())
    response = client.post(
        "/api/chat",
        json={
            "task": "qa",
            "style": "formal",
            "text": "when was it released?",
            "history": [],
            "context_paths": ["alice.md"],
        },
    )
    assert response.status_code == 200, response.text
    assert "Alice was released in 2010" in captured["context"][0]["content"]
    assert captured["context"][0]["name"] == "alice.md"


def test_qa_prompt_includes_context(monkeypatch):
    import src.tasks.qa as qa_module
    from src.tasks import get_task

    monkeypatch.setattr(qa_module, "retrieve_from", lambda *args, **kwargs: "")
    task = get_task("qa", "formal")
    prompt = task.build_prompt(
        "when was it released",
        history=[],
        context=[{"name": "alice.md", "content": "Alice was released in 2010"}],
    )
    assert "Alice was released in 2010" in prompt
    assert "CURRENT DOCUMENT" in prompt
