from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    task: str
    style: str = "formal"
    text: str
    history: list[ChatMessage] = Field(default_factory=list)


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
