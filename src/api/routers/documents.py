import asyncio
import os

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from src.api.deps import embeddings
from src.api.schemas import (
    DeleteResponse,
    DocumentOut,
    UploadResponse,
    WorkspaceDocument,
)
from src.config import UPLOAD_MAX_MB
from src.vector_db import documents
from src.workspace import files as workspace_files

router = APIRouter(prefix="/api/documents", tags=["documents"])


def _validate_kind(kind: str) -> None:
    if kind not in documents.DOCUMENT_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown kind '{kind}'. Use one of {list(documents.DOCUMENT_KINDS)}.",
        )


@router.post("", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    kind: str = Query("userdocs"),
    embedding_function=Depends(embeddings),
):
    _validate_kind(kind)
    filename = documents.safe_filename(file.filename or "document")
    if not documents.is_supported(filename):
        raise HTTPException(
            status_code=400, detail=f"Unsupported file type: {filename}"
        )

    data = await file.read()
    if len(data) > UPLOAD_MAX_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"File exceeds the {UPLOAD_MAX_MB} MB limit."
        )

    path = await asyncio.to_thread(documents.save_upload, kind, filename, data)
    try:
        result = await asyncio.to_thread(
            documents.ingest_document, path, kind, embedding_function
        )
    except Exception as e:
        await asyncio.to_thread(documents.remove_file, path)
        raise HTTPException(
            status_code=422, detail=f"Could not convert or index the file: {e}"
        ) from e

    return UploadResponse(
        doc_id=documents.doc_id_for(path),
        filename=filename,
        kind=kind,
        added=result["added"],
        skipped=result["skipped"],
        total=result["total"],
    )


@router.get("", response_model=list[DocumentOut])
def list_documents(kind: str | None = None, embedding_function=Depends(embeddings)):
    if kind is not None:
        _validate_kind(kind)
    return documents.list_documents(kind, embedding_function=embedding_function)


@router.post("/from-workspace", response_model=UploadResponse)
async def index_workspace_document(
    request: WorkspaceDocument,
    embedding_function=Depends(embeddings),
):
    _validate_kind(request.kind)
    try:
        path = workspace_files.resolve(request.path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"'{request.path}' does not exist in the workspace.")
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        result = await asyncio.to_thread(
            documents.ingest_document, path, request.kind, embedding_function
        )
    except Exception as e:
        raise HTTPException(
            status_code=422, detail=f"Could not convert or index the file: {e}"
        ) from e

    return UploadResponse(
        doc_id=documents.doc_id_for(path),
        filename=os.path.basename(path),
        kind=request.kind,
        added=result["added"],
        skipped=result["skipped"],
        total=result["total"],
    )


@router.delete("/{doc_id}", response_model=DeleteResponse)
def delete_document(
    doc_id: str,
    delete_file: bool = False,
    embedding_function=Depends(embeddings),
):
    result = documents.delete_document(
        doc_id, delete_file=delete_file, embedding_function=embedding_function
    )
    if result["kind"] is None:
        raise HTTPException(
            status_code=404, detail=f"No document with id '{doc_id}'."
        )
    return result
