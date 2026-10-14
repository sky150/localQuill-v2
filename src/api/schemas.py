from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    task: str
    style: str = "formal"
    text: str
    history: list[ChatMessage] = Field(default_factory=list)
    context_paths: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    task: str
    style: str
    result: str


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    kind: str
    added: int
    skipped: int
    total: int


class DocumentOut(BaseModel):
    doc_id: str
    filename: str
    kind: str
    source: str
    chunks: int
    size_bytes: int


class DeleteResponse(BaseModel):
    doc_id: str
    kind: str | None
    deleted_chunks: int
    file_deleted: bool


class HealthResponse(BaseModel):
    ollama: bool
    chroma_path: str
    collections: dict[str, int]


class TaskInfo(BaseModel):
    model: str
    temperature: float
    collection: str | None
    top_k: int


class ConfigResponse(BaseModel):
    styles: list[str]
    tasks: dict[str, TaskInfo]
    upload_extensions: list[str]
    upload_max_mb: int
    guardrails: bool


class WorkspaceDocument(BaseModel):
    path: str
    kind: Literal["userdocs", "outline"] = "outline"


class FileNode(BaseModel):
    name: str
    path: str
    type: Literal["file", "folder"]
    children: list["FileNode"] | None = None


class FileCreate(BaseModel):
    parent: str = ""
    name: str
    kind: Literal["file", "folder"] = "file"


class FileRename(BaseModel):
    path: str
    name: str


class FileMove(BaseModel):
    path: str
    parent: str = ""


class FileContent(BaseModel):
    path: str
    content: str


class PathResponse(BaseModel):
    path: str


class ContentResponse(BaseModel):
    path: str
    content: str


FileNode.model_rebuild()
