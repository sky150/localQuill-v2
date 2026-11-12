import asyncio
import logging
import os

import httpx
from fastapi import APIRouter, HTTPException
from ollama import ResponseError

from src.api.schemas import ChatRequest, ChatResponse, Notice
from src.config import STYLE_COLLECTIONS, TASKS
from src.tasks import get_task
from src.guardrails import screen_attachments, screen_history, screen_user_text
from src.guardrails.notices import collect_notices
from src.tasks.utility import clip
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
    return screen_attachments(context)   # scans what the model would receive


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
        verdict = await asyncio.to_thread(screen_user_text, text)
        if not verdict.allowed:
            raise HTTPException(status_code=400, detail=verdict.message)

    task = get_task(request.task, style=request.style)

    with collect_notices() as notices:
        history = [message.model_dump() for message in request.history]
        # Scanning runs the classifier, so keep it off the event loop.
        history = await asyncio.to_thread(screen_history, history)
        context = await asyncio.to_thread(_load_context, request.context_paths)

        if not text and not context:
            # Everything the user attached was left out; say why instead of running on nothing.
            reasons = " ".join(n["message"] for n in notices) or "Text is empty."
            raise HTTPException(status_code=422, detail=reasons)

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

    return ChatResponse(
        task=request.task,
        style=request.style,
        result=result,
        notices=[Notice(**n) for n in notices],
    )
