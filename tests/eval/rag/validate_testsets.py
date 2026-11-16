"""Check that every gold snippet in the RAG test sets exists in its source file.

Files are converted the same way the app ingests them (markitdown), so a snippet that
passes here is also present in the indexed text. This only proves the snippet exists;
whether it really answers the question still needs a human check.

    uv run python tests/eval/rag/validate_testsets.py
"""

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.vector_db.load_documents import load_file  # noqa: E402

HERE = Path(__file__).parent
TESTSETS = ["testset_qa.json", "testset_style.json"]

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "—": "-", "–": "-"})


def normalize(text: str) -> str:
    """Same normalisation the eval uses to match snippets against chunks."""
    return re.sub(r"\s+", " ", text.translate(_QUOTES)).strip().lower()


@lru_cache(maxsize=None)
def source_text(path: str) -> str:
    return normalize(load_file(str(ROOT / path)).page_content)


def main() -> int:
    problems = 0
    for name in TESTSETS:
        cases = json.loads((HERE / name).read_text())["cases"]
        ids = [c["id"] for c in cases]
        if len(ids) != len(set(ids)):
            print(f"{name}: duplicate ids")
            problems += 1
        found = 0
        for case in cases:
            for gold in case["gold"]:
                if normalize(gold["snippet"]) in source_text(gold["file"]):
                    found += 1
                else:
                    problems += 1
                    print(f"  MISSING {case['id']}: {gold['snippet'][:70]!r} in {gold['file']}")
        total = sum(len(c["gold"]) for c in cases)
        print(f"{name}: {len(cases)} cases, {found}/{total} gold snippets found")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
