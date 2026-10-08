import chainlit as cl
from src.model_query import query_rag
import logging
import asyncio

logging.basicConfig(level=logging.DEBUG)


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Formal", markdown_description="Academic formal writing feedback"
        ),
        cl.ChatProfile(
            name="Fiction", markdown_description="Creative fiction writing feedback"
        ),
        cl.ChatProfile(
            name="Test", markdown_description="Test one single model"
        ),
    ]


@cl.on_chat_start
async def on_start():
    profile = cl.user_session.get("chat_profile")  # "Essay", "Fiction", or "Test"
    style = profile.lower() if profile else "formal"
    cl.user_session.set("style", style)
    await cl.Message(
        content=f"Sending your text for **{profile}** writing feedback."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    style = cl.user_session.get("style", "formal") # style seems to always be formal?
    response = await asyncio.to_thread(query_rag, message.content, style=style)
    await cl.Message(content=response).send()
