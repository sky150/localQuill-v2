import logging
from src.config import TaskConfig
from src.rag.retriever import retrieve, format_context
from src.tasks.utility import truncate_by_paragraph

logger = logging.getLogger(__name__)


class Task:
    name: str
    template: str | None = None             # prompt with {style_context} {guidelines} {text}
    style_context: dict[str, str] = {}      # style -> role text
    retrieval_query: str = ""               # empty = no retrieval
    max_words: int | None = None            # None = no truncation

    def __init__(self, model, config: TaskConfig, style: str = "essay"):
        self.model = model
        self.config = config
        self.style = style

    def retrieve_context(self, text: str) -> str:
        """Guidelines from the task's Chroma collection, or '' if not used or unavailable."""
        if not (self.config.collection and self.retrieval_query):
            return ""
        query = f"{self.retrieval_query}\n\n{text[:2000]}"
        try:
            results = retrieve(self.config.collection, query, self.config.top_k)
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"[{self.name}] retrieval skipped: {e}")
            return ""
        return format_context(results)

    def build_prompt(self, text, **opts) -> str:
        if self.template is None:
            raise NotImplementedError(f"{type(self).__name__} needs a template or its own build_prompt")
        return self.template.format(
            style_context=self.style_context.get(self.style, ""),
            guidelines=self.retrieve_context(text),
            text=text,
        )

    def run(self, text, **opts) -> str:
        was_cut = False
        if self.max_words:
            text, was_cut = truncate_by_paragraph(text, self.max_words)
            logger.info(f"[{self.name}] words sent: {len(text.split())} | cut: {was_cut}")
            
        prompt = self.build_prompt(text, **opts)
        logger.info(f"[{self.name}] prompt chars: {len(prompt)}")
        response = self.model.invoke(prompt)
        
        result = getattr(response, "content", response)
        if was_cut:
            result += f"\n\n---\nNote: input was cut at about {self.max_words} words. Later text was not processed."
        return result