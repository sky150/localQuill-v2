import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.workspace import files


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(files, "WORKSPACE_PATH", str(tmp_path / "workspace"))
    with TestClient(app) as test_client:
        yield test_client


def test_file_lifecycle(client):
    created = client.post(
        "/api/files", json={"parent": "", "name": "Manuscripts", "kind": "folder"}
    )
    assert created.status_code == 200, created.text
    assert created.json()["path"] == "Manuscripts"

    created = client.post(
        "/api/files",
        json={"parent": "Manuscripts", "name": "Chapter.md", "kind": "file"},
    )
    assert created.json()["path"] == "Manuscripts/Chapter.md"

    saved = client.put(
        "/api/files/content",
        json={"path": "Manuscripts/Chapter.md", "content": "# Chapter\n\nHello"},
    )
    assert saved.status_code == 200

    content = client.get(
        "/api/files/content", params={"path": "Manuscripts/Chapter.md"}
    ).json()
    assert content["content"] == "# Chapter\n\nHello"

    tree = client.get("/api/files").json()
    assert tree[0]["name"] == "Manuscripts"
    assert tree[0]["children"][0]["name"] == "Chapter.md"

    renamed = client.post(
        "/api/files/rename",
        json={"path": "Manuscripts/Chapter.md", "name": "Chapter 1.md"},
    )
    assert renamed.json()["path"] == "Manuscripts/Chapter 1.md"

    deleted = client.delete("/api/files", params={"path": "Manuscripts"})
    assert deleted.status_code == 200
    assert client.get("/api/files").json() == []


def test_rejects_escape_and_duplicates(client):
    assert (
        client.get("/api/files/content", params={"path": "../secret.md"}).status_code
        == 400
    )
    assert client.delete("/api/files", params={"path": "../"}).status_code == 400

    first = client.post("/api/files", json={"name": "a.md", "kind": "file"})
    assert first.status_code == 200
    duplicate = client.post("/api/files", json={"name": "a.md", "kind": "file"})
    assert duplicate.status_code == 409


def test_move_file_into_folder(client):
    client.post("/api/files", json={"name": "Folder", "kind": "folder"})
    client.post("/api/files", json={"name": "a.md", "kind": "file"})

    moved = client.post("/api/files/move", json={"path": "a.md", "parent": "Folder"})
    assert moved.status_code == 200, moved.text
    assert moved.json()["path"] == "Folder/a.md"

    tree = {node["name"]: node for node in client.get("/api/files").json()}
    assert tree["Folder"]["children"][0]["name"] == "a.md"

    into_itself = client.post(
        "/api/files/move", json={"path": "Folder", "parent": "Folder"}
    )
    assert into_itself.status_code == 400

    client.post("/api/files", json={"name": "a.md", "kind": "file"})
    collision = client.post(
        "/api/files/move", json={"path": "a.md", "parent": "Folder"}
    )
    assert collision.status_code == 409

    client.delete("/api/files", params={"path": "a.md"})
    back_to_root = client.post(
        "/api/files/move", json={"path": "Folder/a.md", "parent": ""}
    )
    assert back_to_root.status_code == 200
    assert back_to_root.json()["path"] == "a.md"
