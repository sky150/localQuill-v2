from .base import Task
from src.rag.retriever import retrieve_from
from .utility import clip, format_attachments


class QATask(Task):
    name = "qa"
    max_words = None
    collections = {"userdocs": 3, "outline": 1}

    style_context = {
        "formal": "You answer questions about the user's documents and academic writing.",
        "fiction": "You answer questions about the user's documents and fiction writing.",
    }

    template = """ROLE: {style_context}

CONVERSATION SO FAR:
{history}

CURRENT DOCUMENT (the user's own writing, attached by the user):
{current_documents}

USER OUTLINE:
{outline}

USER DOCUMENTS:
{userdocs}

QUESTION:
{text}

RULES:
- Write in en-GB
- Answer from CURRENT DOCUMENT, USER OUTLINE and USER DOCUMENTS. If none contain the answer, say so
- When asked whether the writing fits the outline or requirements, compare CURRENT DOCUMENT with USER OUTLINE and USER DOCUMENTS
- Do not invent facts about the documents
"""

    def build_prompt(self, text, history=None, context=None, **opts) -> str:
        """Override with multiple RAG collections for Question answering."""
        
        outline = clip(retrieve_from(self, "outline", text, top_k=self.collections["outline"]), 800)
        # Results are bad when files are short and fewer than three. Since duplication will happen.
        userdocs = clip(retrieve_from(self, "userdocs", text, top_k=self.collections["userdocs"]), 3000)
        current_documents = clip(format_attachments(context), 4000) if context else ""
        
        return self.template.format(
            style_context=self.style_context.get(self.style, ""),
            outline=outline or "none found",
            userdocs=userdocs or "none found",
            current_documents=current_documents or "none attached",
            history=self._format_history(history),
            text=text,
        )

    @staticmethod
    def _format_history(history, max_chars=1500, per_message=400) -> str:
        """Format the conversation history for inclusion in the prompt."""
        if not history:
            return "none"
        lines, total = [], 0
        for m in reversed(history):     # newest first
            if m.get("task") != "qa":
                continue
            line = f"{m['role']}: {m['content'][:per_message]}"
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line)
        return "\n".join(reversed(lines)) or "none"