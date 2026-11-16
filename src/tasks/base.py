import logging
import time
from pathlib import Path
from src.config import TaskConfig
from src.guardrails.containment import LEAK_MESSAGE, containment_note, leaked, new_canary
from src.guardrails.notices import report_text
from src.rag.retriever import format_references, retrieve_reranked
from src.tasks.utility import format_attachments, truncate_by_paragraph
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

# Hand-picked guide passages per task and style ({task}_{style}.md). Always in the prompt, so the
# core rules are there even when the search picks the wrong passage (docs/rag-evaluation.md).
GUIDELINES_DIR = Path(__file__).parent / "guidelines"

# Search query for the style guides. The model picks the text's main problem from a fixed list of
# guide topics (Task.problem_labels) and that label is the search query. A query built from the text
# finds passages about its topic; a free-text query written by a small model often names the wrong
# problem; a label finds the right passage far more often (docs/rag-evaluation.md, Finding 6).
STYLE_QUERY_PROMPT = """You help look up advice in a writing style guide.
Which {focus} problem is the most serious one in the draft? Choose exactly 1 from the LIST.
Copy the chosen item exactly. Reply with the list item only.

LIST:
{labels}

DRAFT:
\"\"\"
{draft}
\"\"\""""


def pick_label(reply: str, labels: list[str]) -> str:
    """The first list item named in the model's reply, or '' if it names none."""
    for line in reply.lower().splitlines():
        line = line.strip(" -*\"'.0123456789)")
        for label in labels:
            if line and (label.lower() in line or line in label.lower()):
                return label
    return ""


def join_guidelines(fixed: str, retrieved: str) -> str:
    return "\n\n---\n\n".join(part for part in (fixed, retrieved) if part)


class Task:
    name: str
    template: str | None = None             # prompt with {style_context} {guidelines} {text}
    style_context: dict[str, str] = {}      # style -> role text
    retrieval_query: str = ""               # empty = no retrieval; fallback search query
    query_focus: str = ""                   # what the style query looks for (style_query)
    problem_labels: list[str] = []          # guide topics the model picks from (style_query)
    max_words: int | None = None            # None = no truncation
    uses_context_as_text = False            # attached files replace the text to review
    contain_untrusted = False               # add the security note + canary (not for autocomplete)

    def __init__(self, model, config: TaskConfig, style: str = "formal"):
        self.model = model
        self.config = config
        self.style = style
        self.chain = model | StrOutputParser()
                        # Build the chain with the prompt, model, and output parser
                        # Pipe prompt, since langchain allows chaining the prompt through the model and output parser
                    

    def fixed_guidelines(self) -> str:
        """Hand-picked guidelines for this task and style, or '' if there is no file."""
        path = GUIDELINES_DIR / f"{self.name}_{self.style}.md"
        return path.read_text(encoding="utf-8").strip() if path.exists() else ""

    def style_query(self, text: str) -> str:
        """Search query for the style guides: the problem label the task model picks for the text
        (one; a second label mostly added passages for a wrong problem). Falls back to the fixed
        keywords if the model fails or names no label."""
        labels = self.problem_labels
        try:
            reply = self.chain.invoke(STYLE_QUERY_PROMPT.format(
                focus=self.query_focus, labels="\n".join(f"- {l}" for l in labels), draft=text[:2000]))
        except Exception as e:  # the main task should still run without guide chunks
            logger.warning(f"[{self.name}] style query failed, using keywords: {e}")
            return self.retrieval_query
        label = pick_label(reply, labels)
        logger.info(f"[{self.name}] style query: {label or 'no label picked, using keywords'}")
        return label or self.retrieval_query

    def search_guides(self, collection: str, text: str, top_k: int) -> str:
        """Style-guide passages for the text's problem (label query, re-ranked), marked as
        references. '' if the task's search is off (collection=None) or unavailable."""
        if not (self.config.collection and self.retrieval_query):
            return ""
        try:
            return format_references(retrieve_reranked(collection, self.style_query(text), top_k))
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"[{self.name}] retrieval from '{collection}' skipped: {e}")
            return ""

    def retrieve_context(self, text: str) -> str:
        """Fixed guidelines plus style-guide passages from the task's Chroma collection."""
        retrieved = self.search_guides(self.config.collection, text, self.config.top_k)
        return join_guidelines(self.fixed_guidelines(), retrieved)

    def build_prompt(self, text, **opts) -> str:
        if self.template is None:
            raise NotImplementedError(f"{type(self).__name__} needs a template or its own build_prompt")
        return self.template.format(
            style_context=self.style_context.get(self.style, ""),
            guidelines=self.retrieve_context(text),
            text=text,
            note=opts.get("note") or "none",
        )

    def run(self, text, **opts) -> str:
        context = opts.pop("context", None)
        was_cut = False
        if self.uses_context_as_text and context:
            opts["note"] = text.strip()
            text = format_attachments(context)
        if self.max_words:
            text, was_cut = truncate_by_paragraph(text, self.max_words)
            logger.info(f"[{self.name}] words sent: {len(text.split())} | cut: {was_cut}")
            
        prompt = self.build_prompt(text, context=context, **opts)
        canary = None
        if self.contain_untrusted:
            canary = new_canary()
            prompt += containment_note(canary)
        logger.info(f"[{self.name}] prompt chars: {len(prompt)}")
        
        # Invoke the model
        time_start = time.time()
        result = self.chain.invoke(prompt)
        time_passed = time.time() - time_start
        logger.info(f"[{self.name}] response time: {time_passed:.2f}s")

        if canary and leaked(canary, result):
            logger.warning(f"[{self.name}] canary found in output; reply withheld")
            report_text("output", LEAK_MESSAGE)
            return LEAK_MESSAGE

        if was_cut:
            result += f"\n\n---\nNote: input was cut at about {self.max_words} words. Later text was not processed."
        return result + f"\n\n---\nResponse time: {time_passed:.2f}s"