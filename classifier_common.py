"""Shared constants/helpers for the conditional-type classifier (training + Gradio UI).

The classification task uses the `prompt_variant` field already recorded in the generated
datasets as its label:

- "negative" -- persian_negatives_bio.jsonl: no conditional relation at all.
- "marked"   -- persian_conditionals_bio.jsonl: a conditional using an explicit marker word
                (اگر, هرگاه, در صورتی که, چنانچه, ...).
- "unmarked" -- persian_conditionals_bio.jsonl: a conditional expressed implicitly, without a
                marker word.

Kept in one place so `train_classifier.py` and `classifier_ui.py` can't drift apart on label
order, colors, or the base model name.
"""

import json

MODEL_NAME = "HooshvareLab/bert-fa-zwnj-base"
MAX_LENGTH = 64

# Fixed order: index == label id. Also the color-assignment order below, per the dataviz
# categorical-palette rule (assign hues in fixed order, never cycled by rank).
CLASS_NAMES = ["negative", "marked", "unmarked"]

# Short Persian glosses for the UI.
CLASS_DISPLAY_FA = {
    "negative": "بدون شرط",
    "marked": "شرطی با نشانه (اگر...)",
    "unmarked": "شرطی بدون نشانه",
}

# First three slots of the validated categorical palette (see dataviz skill /
# references/palette.md) -- with exactly 3 classes these clear the all-pairs CVD/contrast
# floors in both light and dark mode, unlike the 15-topic case in parsbert_embeddings.ipynb.
CLASS_COLORS = {
    "negative": "#2a78d6",  # blue
    "marked": "#eb6834",  # orange
    "unmarked": "#1baf7a",  # aqua
}

LABEL2ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID2LABEL = {i: name for i, name in enumerate(CLASS_NAMES)}


def load_jsonl(path):
    """Read a .jsonl file into a list of dicts, skipping blank lines."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_labeled_texts(positives_path, negatives_path):
    """Load and merge both dataset files into a list of {"text", "label"} dicts, with `label`
    taken from each example's `prompt_variant` field (see module docstring for the mapping)."""
    records = load_jsonl(positives_path) + load_jsonl(negatives_path)
    labeled = []
    for ex in records:
        variant = ex.get("prompt_variant")
        if variant not in LABEL2ID:
            raise ValueError(
                f"Unexpected prompt_variant {variant!r} (expected one of {CLASS_NAMES}) "
                f"in example: {ex.get('id')!r}"
            )
        labeled.append({"text": ex["text"], "label": LABEL2ID[variant]})
    return labeled
