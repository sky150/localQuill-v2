import re
import unicodedata

# Characters that render as nothing (or reorder text) but still sit inside the string.
# Written as \u escapes so they are visible here. Each line is one group:
_INVISIBLE = re.compile(
    r"["
    r"​-‏"  # zero-width space/joiners, left-to-right / right-to-left marks
    r"‪-‮"  # bidi embeddings and overrides
    r"⁠-⁤"  # word joiner and other invisible operators
    r"⁦-⁩"  # bidi isolates
    r"﻿"         # zero-width no-break space (byte order mark)
    r"­"         # soft hyphen
    r"]"
)
# C0/C1 control characters except tab, newline and carriage return.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def normalize(text: str) -> str:
    """Make text comparable before scanning.

    Attackers can hide a phrase from the patterns by inserting invisible characters
    ("ig<zero-width space>nore previous instructions" looks identical on screen but
    does not match a regex for "ignore"). NFKC folds look-alike forms (full-width
    letters, ligatures) to plain ones, and the two regexes above strip the invisible
    and control characters. The original text is what goes to the model; this
    result is only used for detection.
    """
    text = unicodedata.normalize("NFKC", text) # f.e ｉｇｎｏｒｅ becomes ignore
    text = _INVISIBLE.sub("", text)
    return _CONTROL.sub("", text)
