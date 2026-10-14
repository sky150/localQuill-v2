import asyncio
import logging
import os

import httpx
from fastapi import APIRouter, HTTPException
from ollama import ResponseError

from src.api.schemas import ChatRequest, ChatResponse
from src.config import STYLE_COLLECTIONS, TASKS
from src.tasks import get_task
from src.tasks.utility import clip, use_guardrails
from src.workspace import files

logger = logging.getLogger(__name__)

CONTEXT_CHAR_LIMIT = 4000
MAX_CONTEXT_FILES = 3

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _load_context(paths: list[str]) -> list[dict]:
    if not paths:
        return []
    context = []
    for path in paths[:MAX_CONTEXT_FILES]:
        try:
            content = files.read_text(path)
        except (ValueError, FileNotFoundError) as error:
            logger.warning(f"Attached file skipped: {error}")
            continue
        context.append(
            {
                "name": os.path.basename(path),
                "path": path,
                "content": clip(content, CONTEXT_CHAR_LIMIT),
            }
        )
    return context


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if request.task not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{request.task}'.")
    if request.style not in STYLE_COLLECTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown style '{request.style}'.")

    text = request.text.strip()
    if not text and not request.context_paths:
        raise HTTPException(status_code=422, detail="Text is empty.")

    if text:
        blocked = use_guardrails(text)
        if blocked:
            raise HTTPException(status_code=400, detail=blocked)

    task = get_task(request.task, style=request.style)
    history = [message.model_dump() for message in request.history]
    context = _load_context(request.context_paths)

    try:
        result = await asyncio.to_thread(
            task.run, text, history=history, context=context
        )
    except NotImplementedError as e:
        raise HTTPException(
            status_code=400, detail=f"Task '{request.task}' is not implemented yet."
        ) from e
    except (httpx.ConnectError, httpx.ConnectTimeout, ResponseError) as e:
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}") from e

    return ChatResponse(task=request.task, style=request.style, result=result)
