import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import chat, documents, files, meta

app = FastAPI(title="Local Quill API", version="0.1.0")

app.include_router(meta.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(files.router)

_origins = [
    origin.strip()
    for origin in os.getenv("API_CORS_ORIGINS", "").split(",")
    if origin.strip()
]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
