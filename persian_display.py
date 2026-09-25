"""Utilities for displaying Persian text cleanly.

Persian text often contains the zero-width non-joiner (ZWNJ, U+200C), used inside
compound words like "می‌خواهم" (mi-khāham) to join prefix/suffix without a visible
space while keeping the letters unconnected. It's invisible when rendered with a
font that supports it, but two things commonly go wrong when inspecting such text:

1. The string was round-tripped through something that escaped it as the literal
   4 characters ``\\u200c`` (backslash, u, 2, 0, 0, c) instead of the real character
   -- e.g. ``json.dumps(text)`` with the default ``ensure_ascii=True``, or a naive
   ``repr()``/logging call. Printing it then shows the literal escape sequence.
2. The terminal/font doesn't render the real ZWNJ character well, so it shows up
   as a visible box, gap, or is otherwise distracting when just eyeballing text.

``display_persian`` handles both: it decodes any literal ``\\u200c``-style escapes
back into a real character, then (by default) strips ZWNJ entirely for a clean,
readable string.
"""

import re
import sys

ZWNJ = "‌"

# Matches a literal backslash-u escape for ZWNJ as it appears in raw text, e.g. from
# json.dumps(..., ensure_ascii=True) or repr(): the four characters \, u, 2, 0, 0, c.
_LITERAL_ZWNJ_ESCAPE = re.compile(r"\\u200[cC]")


def decode_literal_escapes(text):
    """Turn a literal ``\\u200c`` text escape into the real ZWNJ character.

    No-op if the string already contains real ZWNJ characters (or none at all).
    """
    return _LITERAL_ZWNJ_ESCAPE.sub(ZWNJ, text)


def clean_persian_text(text, replace_zwnj_with=""):
    """Return `text` with ZWNJ characters normalized for clean display.

    Decodes any literal ``\\u200c`` escapes first (see `decode_literal_escapes`),
    then replaces every real ZWNJ character with `replace_zwnj_with` (default: "",
    i.e. removed). Pass ``replace_zwnj_with=" "`` if you'd rather see a visible
    space where words were joined.
    """
    text = decode_literal_escapes(text)
    return text.replace(ZWNJ, replace_zwnj_with)


def display_persian(text, replace_zwnj_with=""):
    """Print `text` with ZWNJ escapes/characters cleaned up for easy reading.

    Also works around Windows consoles that default to a non-Unicode codepage
    (e.g. cp1252), which otherwise raise UnicodeEncodeError on Persian text:
    falls back to writing UTF-8 bytes directly to stdout's buffer.
    """
    cleaned = clean_persian_text(text, replace_zwnj_with=replace_zwnj_with)
    try:
        print(cleaned)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(cleaned.encode("utf-8") + b"\n")


if __name__ == "__main__":
    samples = [
        "این روزا تو ایران همه دارن از سیاست حرف می‌زنن .",  # real ZWNJ
        "می\\u200cخواهم به مدرسه بروم.",  # literal escape text
    ]
    for s in samples:
        display_persian(f"cleaned: {clean_persian_text(s)}")
