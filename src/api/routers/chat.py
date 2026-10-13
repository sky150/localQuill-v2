import asyncio

import httpx
from fastapi import APIRouter, HTTPException
from ollama import ResponseError

from src.api.schemas import ChatRequest, ChatResponse
from src.config import STYLE_COLLECTIONS, TASKS
from src.tasks import get_task
from src.tasks.utility import use_guardrails

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if request.task not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{request.task}'.")
    if request.style not in STYLE_COLLECTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown style '{request.style}'.")

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text is empty.")

    blocked = use_guardrails(text)
    if blocked:
        raise HTTPException(status_code=400, detail=blocked)

    task = get_task(request.task, style=request.style)
    history = [message.model_dump() for message in request.history]

    try:
        result = await asyncio.to_thread(task.run, text, history=history)
    except NotImplementedError as e:
        raise HTTPException(
            status_code=400, detail=f"Task '{request.task}' is not implemented yet."
        ) from e
    except (httpx.ConnectError, httpx.ConnectTimeout, ResponseError) as e:
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}") from e

    return ChatResponse(task=request.task, style=request.style, result=result)
