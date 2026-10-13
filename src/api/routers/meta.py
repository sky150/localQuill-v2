import os

import httpx
from fastapi import APIRouter

from src.api.schemas import ConfigResponse, HealthResponse, TaskInfo
from src.config import (
    CHROMA_PATH,
    OLLAMA_URL,
    STYLE_COLLECTIONS,
    TASKS,
    UPLOAD_MAX_MB,
    get_config,
)
from src.vector_db.documents import DOCUMENT_KINDS, open_collection
from src.vector_db.load_documents import SUPPORTED_EXTENSIONS

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/health", response_model=HealthResponse)
def health():
    ollama_up = False
    try:
        response = httpx.get(f"{OLLAMA_URL}/api/version", timeout=2.0)
        ollama_up = response.status_code == 200
    except httpx.HTTPError:
        pass

    collections = {}
    if os.path.exists(CHROMA_PATH):
        for name in ("formal", "fiction", *DOCUMENT_KINDS):
            collections[name] = len(open_collection(name).get(include=[])["ids"])

    return HealthResponse(
        ollama=ollama_up,
        chroma_path=CHROMA_PATH,
        collections=collections,
    )


@router.get("/config", response_model=ConfigResponse)
def config():
    tasks = {}
    for name in TASKS:
        task_config = get_config(name)
        tasks[name] = TaskInfo(
            model=task_config.model,
            temperature=task_config.temperature,
            collection=task_config.collection,
            top_k=task_config.top_k,
        )
    return ConfigResponse(
        styles=list(STYLE_COLLECTIONS),
        tasks=tasks,
        upload_extensions=sorted(SUPPORTED_EXTENSIONS),
        upload_max_mb=UPLOAD_MAX_MB,
        guardrails=os.getenv("USE_GUARDRAILS", "true").strip().lower() != "false",
    )
