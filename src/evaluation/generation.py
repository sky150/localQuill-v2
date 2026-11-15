"""Generation through the app's Task classes and model factory.

build_task makes the app's task (template) with the model from src.llm.factory.get_model, so
the tested model is the one the app would build. generate does not call Task.run(): run()
appends a random canary (the prompt would differ per call and per model), truncates at
max_words, appends a "Response time" trailer, and its StrOutputParser drops the token counts.
Instead generate calls task.model.invoke and keeps the AIMessage metadata.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.config import TaskConfig
from src.llm.factory import get_model
from src.tasks.base import Task

from . import config
from .common import TASK_BY_PROMPT, EvalPaths

NS = 1e9


class UnknownProvider(SystemExit):
    """The provider has no generation options mapping in this module."""


@dataclass(frozen=True)
class Generation:
    raw: str
    wall_s: float
    load_s: float | None
    prompt_tokens: int | None
    gen_tokens: int | None
    gen_tok_per_s: float | None
    done_reason: str | None


def build_task(paths: EvalPaths, provider: str, model: str) -> Task:
    task_cls = TASK_BY_PROMPT[paths.prompt]
    cfg = TaskConfig(model=model, collection=paths.style, top_k=config.RAG_TOP_K,
                     max_tokens=config.NUM_PREDICT)
    llm = get_model(task_cls.name, paths.style, cfg=cfg, provider=provider)
    return task_cls(model=llm, config=cfg, style=paths.style)


def _invoke_kwargs(provider: str, params: dict, seed: int | None, num_predict: int) -> dict:
    if provider == "ollama":
        # `options` replaces every option the factory set (its num_ctx=4096 included), so the
        # eval's context size, cap and seed apply.
        return {"options": {**params, "num_ctx": config.NUM_CTX, "num_predict": num_predict,
                            **({"seed": seed} if seed is not None else {})}}
    if provider == "lmstudio":
        return {**params, "max_tokens": num_predict, **({"seed": seed} if seed is not None else {})}
    raise UnknownProvider(f"Unknown provider {provider!r}; choose one of {config.PROVIDERS}.")


def generate(task: Task, provider: str, prompt: str, params: dict, seed: int) -> Generation:
    kwargs = _invoke_kwargs(provider, params, seed, config.NUM_PREDICT)
    t0 = time.perf_counter()
    msg = task.model.invoke(prompt, **kwargs)
    wall_s = time.perf_counter() - t0
    meta = msg.response_metadata or {}
    usage = msg.usage_metadata or {}
    gen_tokens = usage.get("output_tokens")
    eval_s = (meta.get("eval_duration") or 0) / NS
    load = meta.get("load_duration")
    return Generation(
        raw=msg.content, wall_s=wall_s,
        load_s=load / NS if load is not None else None,
        prompt_tokens=usage.get("input_tokens"), gen_tokens=gen_tokens,
        # Only Ollama reports decode time; LM Studio's OpenAI API does not, and wall time
        # includes prompt processing, so no rate is reported there.
        gen_tok_per_s=gen_tokens / eval_s if gen_tokens and eval_s else None,
        done_reason=meta.get("done_reason") or meta.get("finish_reason"))


def warm_up(task: Task, provider: str) -> None:
    """Load the model once so load time does not land in run 0's timing."""
    task.model.invoke("Hi", **_invoke_kwargs(provider, {}, None, 1))
