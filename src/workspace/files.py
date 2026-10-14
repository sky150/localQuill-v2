import os
import shutil

from src.config import WORKSPACE_PATH

EDITABLE_EXTENSIONS = {".md", ".markdown", ".txt"}


def workspace_root() -> str:
    root = os.path.realpath(os.path.abspath(WORKSPACE_PATH))
    os.makedirs(root, exist_ok=True)
    return root


def resolve(relative: str) -> str:
    root = workspace_root()
    target = os.path.realpath(os.path.join(root, relative)) if relative else root
    if target != root and not target.startswith(root + os.sep):
        raise ValueError("Path escapes the workspace.")
    return target


def to_relative(path: str) -> str:
    return os.path.relpath(path, workspace_root())


def _node(path: str) -> dict:
    name = os.path.basename(path)
    if os.path.isdir(path):
        children = [_node(os.path.join(path, entry)) for entry in os.listdir(path)]
        children.sort(key=lambda child: (child["type"] != "folder", child["name"].lower()))
        return {
            "name": name,
            "path": to_relative(path),
            "type": "folder",
            "children": children,
        }
    return {"name": name, "path": to_relative(path), "type": "file"}


def build_tree() -> list[dict]:
    root = workspace_root()
    nodes = [_node(os.path.join(root, entry)) for entry in os.listdir(root)]
    nodes.sort(key=lambda node: (node["type"] != "folder", node["name"].lower()))
    return nodes


def validate_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned or cleaned in {".", ".."} or "/" in cleaned or "\\" in cleaned:
        raise ValueError("Invalid name.")
    return cleaned


def create_node(parent: str, name: str, kind: str) -> str:
    parent_path = resolve(parent)
    if not os.path.isdir(parent_path):
        raise FileNotFoundError("Parent folder does not exist.")
    name = validate_name(name)
    target = os.path.join(parent_path, name)
    if os.path.exists(target):
        raise FileExistsError(f"'{name}' already exists.")
    if kind == "folder":
        os.makedirs(target)
    elif kind == "file":
        open(target, "w", encoding="utf-8").close()
    else:
        raise ValueError("Kind must be 'file' or 'folder'.")
    return to_relative(target)


def read_text(path: str) -> str:
    target = resolve(path)
    if not os.path.isfile(target):
        raise FileNotFoundError(f"'{path}' does not exist.")
    if os.path.splitext(target)[1].lower() not in EDITABLE_EXTENSIONS:
        raise ValueError("Only markdown and text files can be opened in the editor.")
    with open(target, encoding="utf-8") as f:
        return f.read()


def write_text(path: str, content: str) -> None:
    target = resolve(path)
    if not os.path.isfile(target):
        raise FileNotFoundError(f"'{path}' does not exist.")
    if os.path.splitext(target)[1].lower() not in EDITABLE_EXTENSIONS:
        raise ValueError("Only markdown and text files can be edited.")
    with open(target, "w", encoding="utf-8") as f:
        f.write(content)


def rename(path: str, name: str) -> str:
    target = resolve(path)
    if not os.path.exists(target):
        raise FileNotFoundError(f"'{path}' does not exist.")
    name = validate_name(name)
    destination = os.path.join(os.path.dirname(target), name)
    if os.path.exists(destination):
        raise FileExistsError(f"'{name}' already exists.")
    os.rename(target, destination)
    return to_relative(destination)


def move(path: str, parent: str) -> str:
    source = resolve(path)
    if not os.path.exists(source):
        raise FileNotFoundError(f"'{path}' does not exist.")
    destination_parent = resolve(parent)
    if not os.path.isdir(destination_parent):
        raise FileNotFoundError("Destination folder does not exist.")
    if os.path.isdir(source) and (
        destination_parent == source
        or destination_parent.startswith(source + os.sep)
    ):
        raise ValueError("Cannot move a folder into itself.")
    destination = os.path.join(destination_parent, os.path.basename(source))
    if destination == source:
        return to_relative(source)
    if os.path.exists(destination):
        raise FileExistsError(f"'{os.path.basename(source)}' already exists there.")
    os.rename(source, destination)
    return to_relative(destination)


def delete(path: str) -> None:
    target = resolve(path)
    if target == workspace_root():
        raise ValueError("Cannot delete the workspace root.")
    if os.path.isdir(target):
        shutil.rmtree(target)
    elif os.path.isfile(target):
        os.remove(target)
    else:
        raise FileNotFoundError(f"'{path}' does not exist.")
