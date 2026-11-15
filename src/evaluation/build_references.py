"""Build the LanguageTool references and report which hand-written ones are missing.

    uv run python -m src.evaluation.build_references --style formal --prompt grammar
    uv run python -m src.evaluation.build_references --style formal --prompt grammar --lt-public

- languagetool: applies LanguageTool's first suggestion per match.
  Existing files are never overwritten, so hand-corrected LT files stay.
  Local mode needs Java; --lt-public uses the rate-limited public API
  and pauses config.LT_PUBLIC_DELAY_S (10 s) after each text.
- opus: written by hand in a separate preparation step, one file per test item,
  following the same rules as the app's grammar prompt. This script only
  reports which files are missing. data/eval/<style>/opus/ is written ahead
  of time; data/eval/<style>/raw holds source files still to prepare and
  data/eval/<style>/error_texts holds the texts with errors.
"""

from __future__ import annotations

import argparse
import time

from . import config
from .common import EvalPaths, add_target_args, load_testset, paths_from_args


def build_languagetool(paths: EvalPaths, items: list[dict], public: bool) -> None:
    import language_tool_python as ltp

    out_dir = paths.reference_dir("languagetool")
    out_dir.mkdir(parents=True, exist_ok=True)
    tool = ltp.LanguageToolPublicAPI(config.LT_LANGUAGE) if public else ltp.LanguageTool(config.LT_LANGUAGE)
    try:
        for item in items:
            path = out_dir / f"{item['id']}.txt"
            if path.exists():
                print(f"[lt] skip {item['id']} (exists)")
                continue
            path.write_text(tool.correct(item["text"]), encoding="utf-8")
            print(f"[lt] wrote {item['id']}")
            if public:
                time.sleep(config.LT_PUBLIC_DELAY_S)
    finally:
        tool.close()


def report_missing(paths: EvalPaths, items: list[dict]) -> None:
    for name in config.REFERENCES:
        if name == "languagetool":
            continue
        missing = [it["id"] for it in items if not (paths.reference_dir(name) / f"{it['id']}.txt").exists()]
        print(f"[{name}] missing: {', '.join(missing)}" if missing else f"[{name}] all present")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lt-public", action="store_true")
    add_target_args(ap)
    args = ap.parse_args()
    paths = paths_from_args(args)

    items = load_testset(paths)
    build_languagetool(paths, items, args.lt_public)
    report_missing(paths, items)


if __name__ == "__main__":
    main()
