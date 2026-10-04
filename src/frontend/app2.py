import asyncio
import chainlit as cl
from chainlit.input_widget import Select
from src.config import TASKS
from src.tasks import get_task  # your dict: name -> task instance

MAX_TURNS = 3  # last 3 messages, keeps local 4b models inside their context
# $env:PYTHONPATH="."; uv run chainlit run src/frontend/app2.py

@cl.on_chat_start
async def on_start():
    settings = await cl.ChatSettings([
        Select(id="style", label="Style", values=["essay", "fiction"], initial_index=1),
        Select(id="task", label="Task", values=list(TASKS), initial_index=2),       # Test, Grammar, Q&A, Feedback, Autocorrect
    ]).send()
    cl.user_session.set("settings", settings)
    cl.user_session.set("history", [])

@cl.on_settings_update
async def on_update(settings):
    cl.user_session.set("settings", settings)

@cl.on_message
async def main(message: cl.Message):
    s = cl.user_session.get("settings")
    history = cl.user_session.get("history")
    task = get_task(s["task"], style=s["style"])

    response = await asyncio.to_thread(
        task.run, message.content, style=s["style"], history=history[-MAX_TURNS:]
    )

    history.append({"role": "user", "task": s["task"], "content": message.content})
    history.append({"role": "assistant", "task": s["task"], "content": response})
    await cl.Message(content=response).send()