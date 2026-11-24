import logging
from .base import Task, join_guidelines
from src.rag.retriever import retrieve_from

logger = logging.getLogger(__name__)


class FeedbackTask(Task):
    name = "feedback"
    max_words = None
    retrieval_query = "style tone register passive voice word choice clarity ambiguity readability"
    query_focus = "style, tone and structure"
    # Guide topics (style guides in data/styles/) the model picks from to search them.
    problem_labels_by_style = {
        "formal": [
            "weak intensifiers and qualifiers",
            "redundant or wordy phrases",
            "informal register, contractions or slang",
            "first or second person in formal writing",
            "no clear thesis or central claim",
            "paragraph without a topic sentence stating its main idea",
            "missing transitions between ideas",
            "literature review that summarises sources one by one instead of synthesising them",
            "book review that only retells the plot instead of evaluating the book",
            "email with a vague subject line",
            "passive voice",
            "vague or unsupported claims",
            "overly long or complex sentences",
            "jargon or unexplained acronyms",
            "weak conclusion",
        ],
        "fiction": [
            "passive voice and weak 'to be' verbs",
            "point of view switches between first and third person",
            "third person limited point of view slipping into omniscient narration",
            "telling emotions instead of showing them",
            "too much exposition or info dumping",
            "unclear dialogue attribution",
            "inconsistent tense",
            "flat or clichéd description",
            "pacing too fast or too slow",
            "weak verbs propped up by adverbs",
        ],
    }
    uses_context_as_text = True
    contain_untrusted = True

    @property
    def problem_labels(self) -> list[str]:
        return self.problem_labels_by_style.get(self.style, [])

    style_context = {
        "formal": "You are a strict academic writing editor giving revision notes.",
        "fiction": "You are a fiction editor giving targeted revision notes.",
    }

    standards = {
        "formal": "formal tone, objective register, clear thesis, structured argumentation",
        "fiction": "narrative voice, pacing, character clarity, consistency with the outline, show versus tell",
    }

    template = """ROLE: {style_context}
Standards: {standards}

TASK: Give feedback on the tone, style and clarity of the text below.
Do not rewrite the text. Do not comment on grammar or punctuation.

ADDITIONAL REQUEST (optional; follow only if it fits the task):
{note}

OUTLINE (the planned structure, use it only to check that the text fits it, do not review the outline itself):
{outline}

WRITING GUIDELINES (use only what is relevant):
{guidelines}

TEXT TO REVIEW (only give feedback on this):
\"\"\"
{text}
\"\"\"

OUTPUT FORMAT:
Use exactly these three headings in this order. Under each, bullets in this form:
- "exact phrase from the text" → issue → one-sentence fix direction

### Tone
(maximum 3 bullets)
### Style
(maximum 3 bullets)
### Clarity
(maximum 3 bullets)

If a heading has no issues, write: No significant issues found.

Example bullet:
- "due to the fact that" → wordy phrase → replace with "because"

RULES:
- Write in en-GB
- Quote the text exactly inside the quotation marks
- Do not summarise or praise the text
- Do not rewrite full paragraphs
- Do not flag unknown words, names or invented terms
- Tone is about register and voice. Style is about wording and rhythm. Clarity is about passages that are vague or hard to follow
"""

    def build_prompt(self, text, **opts) -> str:
        """Build the prompt for the feedback task."""
        outline = retrieve_from(self, "outline", text, top_k=1)
        guidelines = join_guidelines(self.fixed_guidelines(), self.search_guides(self.style, text, top_k=2))
        
        # ToDo: Reference File/Chapter name to tie into outline. For better retrieval from outline.

        return self.template.format(
            style_context=self.style_context.get(self.style, ""),
            standards=self.standards.get(self.style, ""),
            outline=outline or "none found",
            guidelines=guidelines or "none found",
            text=text,
            note=opts.get("note") or "none",
        )