"""Shared constants/helpers for the IF/THEN span tagger (token classification), as opposed to
`classifier_common.py`'s sentence-level negative/marked/unmarked classifier.

This one predicts a BIO tag per token -- which words are the condition (IF) clause and which are
the consequence (THEN) clause -- directly from the `tokens`/`tags` fields already present in both
generated datasets (`persian_conditionals_bio.jsonl` has real IF/THEN spans; every token in
`persian_negatives_bio.jsonl` is tagged `O`, which is useful negative signal for the tagger too).
"""

import re

from classifier_common import MODEL_NAME, load_jsonl  # re-exported for convenience

MAX_LENGTH = 64

BIO_TAGS = ["O", "B-IF", "I-IF", "B-THEN", "I-THEN"]
BIO_LABEL2ID = {tag: i for i, tag in enumerate(BIO_TAGS)}
BIO_ID2LABEL = {i: tag for i, tag in enumerate(BIO_TAGS)}

# First two categorical slots (see classifier_common.CLASS_COLORS / dataviz palette) -- IF/THEN
# is a 2-category encoding, well within the all-pairs-safe cap.
SPAN_COLORS = {"IF": "#2a78d6", "THEN": "#eb6834"}  # blue, orange

# Same regex tokenizer used by generate_conditionals.ipynb / generate_negatives.ipynb, so word
# boundaries at inference time match the ones the "tokens"/"tags" training data were built with.
TOKEN_PATTERN = re.compile(r"[\w‌]+|[^\s\w]", re.UNICODE)


def tokenize_with_offsets(text):
    """Return a list of (token, start, end) for `text`."""
    return [(m.group(), m.start(), m.end()) for m in TOKEN_PATTERN.finditer(text)]


def load_token_tag_examples(paths):
    """Load {"tokens", "tags", "prompt_variant"} records from one or more .jsonl files.

    `prompt_variant` is carried along only to stratify the train/val/test split (so it stays
    balanced across marked / unmarked / negative examples); it is not a model input or target.
    """
    records = []
    for path in paths:
        for ex in load_jsonl(path):
            if len(ex["tokens"]) != len(ex["tags"]):
                raise ValueError(f"tokens/tags length mismatch in example {ex.get('id')!r} of {path}")
            records.append(
                {
                    "tokens": ex["tokens"],
                    "tags": ex["tags"],
                    "prompt_variant": ex.get("prompt_variant", "unknown"),
                }
            )
    return records


def word_tags_to_char_spans(text, tokens_with_offsets, word_tags):
    """Turn per-token BIO tags into a list of (substring, label) run-length-encoded over the full
    original `text` (label is "IF"/"THEN"/None), suitable for gr.HighlightedText -- concatenating
    the substrings reproduces `text` exactly, including the whitespace between tokens."""
    label_per_char = [None] * len(text)
    for (_tok, start, end), tag in zip(tokens_with_offsets, word_tags):
        label = None if tag == "O" else tag.split("-", 1)[1]
        for idx in range(start, end):
            label_per_char[idx] = label

    spans = []
    i = 0
    while i < len(text):
        label = label_per_char[i]
        j = i
        while j < len(text) and label_per_char[j] == label:
            j += 1
        spans.append((text[i:j], label))
        i = j
    return spans


def tags_to_clauses(tokens, tags):
    """Collapse per-token BIO tags into a list of {"label", "text"} clause dicts (merging
    consecutive B-/I- tokens of the same type), e.g. for a debug/inspection view or comparing
    gold vs. predicted spans by exact text."""
    clauses = []
    current_label = None
    current_tokens = []
    for tok, tag in zip(tokens, tags):
        label = None if tag == "O" else tag.split("-", 1)[1]
        if label != current_label:
            if current_label is not None:
                clauses.append({"label": current_label, "text": " ".join(current_tokens)})
            current_label = label
            current_tokens = []
        if label is not None:
            current_tokens.append(tok)
    if current_label is not None:
        clauses.append({"label": current_label, "text": " ".join(current_tokens)})
    return clauses


def word_tags_to_clauses(tokens_with_offsets, word_tags):
    """Same as `tags_to_clauses`, but takes (token, start, end) tuples as produced by
    `tokenize_with_offsets` -- convenience wrapper for callers that have offsets on hand."""
    tokens = [tok for tok, _s, _e in tokens_with_offsets]
    return tags_to_clauses(tokens, word_tags)
