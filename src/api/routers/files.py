from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import (
    ContentResponse,
    FileContent,
    FileCreate,
    FileMove,
    FileNode,
    FileRename,
    PathResponse,
)
from src.workspace import files

router = APIRouter(prefix="/api/files", tags=["files"])


def _http_error(error: Exception) -> HTTPException:
    if isinstance(error, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(error))
    if isinstance(error, FileExistsError):
        return HTTPException(status_code=409, detail=str(error))
    return HTTPException(status_code=400, detail=str(error))


@router.get("", response_model=list[FileNode])
def tree():
    return files.build_tree()


@router.get("/content", response_model=ContentResponse)
def read_content(path: str = Query(...)):
    try:
        return ContentResponse(path=path, content=files.read_text(path))
    except (ValueError, FileNotFoundError) as error:
        raise _http_error(error) from error


@router.put("/content", response_model=PathResponse)
def write_content(request: FileContent):
    try:
        files.write_text(request.path, request.content)
        return PathResponse(path=request.path)
    except (ValueError, FileNotFoundError) as error:
        raise _http_error(error) from error


@router.post("", response_model=PathResponse)
def create(request: FileCreate):
    try:
        return PathResponse(
            path=files.create_node(request.parent, request.name, request.kind)
        )
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        raise _http_error(error) from error


@router.post("/rename", response_model=PathResponse)
def rename(request: FileRename):
    try:
        return PathResponse(path=files.rename(request.path, request.name))
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        raise _http_error(error) from error


@router.post("/move", response_model=PathResponse)
def move(request: FileMove):
    try:
        return PathResponse(path=files.move(request.path, request.parent))
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        raise _http_error(error) from error


@router.delete("", response_model=PathResponse)
def delete(path: str = Query(...)):
    try:
        files.delete(path)
        return PathResponse(path=path)
    except (ValueError, FileNotFoundError) as error:
        raise _http_error(error) from error
