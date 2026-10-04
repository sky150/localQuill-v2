def truncate_by_paragraph(text: str, max_words: int) -> tuple[str, bool]:
    """Keep whole paragraphs until max_words is reached. Returns (text, was_cut)."""
    kept, count = [], 0
    for para in text.split("\n\n"):
        n = len(para.split())
        if kept and count + n > max_words:
            return "\n\n".join(kept), True
        if not kept and n > max_words:
            return " ".join(para.split()[:max_words]), True
        kept.append(para)
        count += n
    return "\n\n".join(kept), False