import os
from .base import Task


class GrammarTask(Task):
    name = "grammar"
    max_words = int(os.getenv("GRAMMAR_MAX_WORDS", "300"))
    retrieval_query = "grammar punctuation tense agreement comma splice sentence structure"

    style_context = {
        "formal": "You are a careful copy editor for academic texts. Keep the formal tone.",
        "fiction": "You are a careful copy editor for fiction. Keep the narrative voice and dialogue style.",
    }

    template = """ROLE: {style_context}

TASK: Correct the complete text below.
Focus: grammar, punctuation, tense agreement, comma splices, sentence structure.

WRITING GUIDELINES:
{guidelines}

TEXT:
\"\"\"
{text}
\"\"\"

RULES:
- Write in en-GB
- Return the complete corrected text, every paragraph, no exceptions
- Change only grammar, punctuation, tense and sentence structure
- Do not change meaning, word choice or names
- Do not correct unknown words or phrases, keep them unchanged
- After the text, add a line "UNKNOWN WORDS:" followed by the list, or "none"
- No explanations, no comments
"""