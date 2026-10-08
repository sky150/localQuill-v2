from .base import Task
from src.rag.retriever import retrieve_from


class QATask(Task):
    name = "qa"
    max_words = None

    style_context = {
        "formal": "You answer questions about the user's documents and academic writing.",
        "fiction": "You answer questions about the user's documents and fiction writing.",
    }

    template = """ROLE: {style_context}

USER DOCUMENTS:
{documents}

WRITING GUIDES:
{guides}

CONVERSATION SO FAR:
{history}

QUESTION:
{text}

RULES:
- Write in en-GB
- Answer from the documents first. If they do not contain the answer, say so
- Do not invent facts about the documents
"""

    def build_prompt(self, text, history=None, **opts) -> str:
        """Override with multiple RAG collections for Question answering."""
        
        # ToDo: Uncomment once we have user documents
        # documents = retrieve_from(self, "user_docs", text, top_k=5)
        documents = ""
        guides = retrieve_from(self, self.style, text, top_k=2)  # collection name = style

        
        return self.template.format(
            style_context=self.style_context.get(self.style, ""),
            documents=documents or "none found",
            guides=guides or "none found",
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