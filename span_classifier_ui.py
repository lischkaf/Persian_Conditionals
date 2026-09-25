"""Gradio UI for the IF/THEN span tagger trained by train_span_classifier.py.

Enter Persian text and the extracted condition clause (IF, blue) and consequence clause (THEN,
orange) are shown as highlighted chunks -- just the clause text itself, not the full sentence with
every token marked up -- as opposed to classifier_ui.py, which only labels the whole sentence.

Usage:
    python span_classifier_ui.py
    python span_classifier_ui.py --adapter-dir checkpoints/parsbert-lora-span-classifier/final

Requirements:
    pip install torch transformers peft gradio
"""

import argparse
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForTokenClassification, AutoTokenizer

from span_classifier_common import (
    BIO_ID2LABEL,
    BIO_LABEL2ID,
    BIO_TAGS,
    MAX_LENGTH,
    MODEL_NAME,
    SPAN_COLORS,
    tokenize_with_offsets,
    word_tags_to_clauses,
)

EXAMPLES = [
    "اگر باران ببارد، فردا به پارک نمی‌رویم.",
    "بخواب، حالت بهتر می‌شه.",
    "امروز برای شام یه املت خوشمزه درست کردم.",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--adapter-dir",
        default="checkpoints/parsbert-lora-span-classifier/final",
        help="Directory saved by train_span_classifier.py (base model + PEFT adapter + tokenizer).",
    )
    parser.add_argument("--base-model", default=MODEL_NAME)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    return parser.parse_args()


def load_model(adapter_dir, base_model_name):
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"{adapter_dir} not found. Run train_span_classifier.py first (it saves the adapter there)."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    if not tokenizer.is_fast:
        raise RuntimeError(f"{base_model_name} did not load a fast (Rust) tokenizer, which word_ids() needs.")

    base_model = AutoModelForTokenClassification.from_pretrained(
        base_model_name,
        num_labels=len(BIO_TAGS),
        id2label=BIO_ID2LABEL,
        label2id=BIO_LABEL2ID,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    return tokenizer, model


def main():
    args = parse_args()
    tokenizer, model = load_model(args.adapter_dir, args.base_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    @torch.no_grad()
    def predict_word_tags(tokens):
        encoded = tokenizer(
            tokens, truncation=True, max_length=MAX_LENGTH, is_split_into_words=True, return_tensors="pt"
        ).to(device)
        logits = model(**encoded).logits[0]
        pred_ids = logits.argmax(dim=-1).tolist()
        word_ids = encoded.word_ids(batch_index=0)

        word_tags = [None] * len(tokens)
        seen_words = set()
        for subword_idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx in seen_words:
                continue
            seen_words.add(word_idx)
            word_tags[word_idx] = BIO_ID2LABEL[pred_ids[subword_idx]]
        return [tag or "O" for tag in word_tags]

    def tag_text(text):
        text = (text or "").strip()
        if not text:
            return []

        tokens_with_offsets = tokenize_with_offsets(text)
        tokens = [tok for tok, _s, _e in tokens_with_offsets]
        if not tokens:
            return []

        word_tags = predict_word_tags(tokens)
        clauses = word_tags_to_clauses(tokens_with_offsets, word_tags)

        if not clauses:
            return [("(no conditional clause detected)", None)]

        highlighted = []
        for i, clause in enumerate(clauses):
            if i > 0:
                highlighted.append(("   ", None))  # separator between clause chunks
            highlighted.append((clause["text"], clause["label"]))

        return highlighted

    with gr.Blocks(title="Persian Conditional Span Tagger") as demo:
        gr.Markdown(
            "# Persian Conditional Span Tagger\n"
            "Enter a Persian sentence. The extracted **IF** clause (condition, blue) and **THEN** "
            "clause (consequence, orange) are shown below — a LoRA adapter fine-tuned on ParsBERT "
            "(`" + args.base_model + "`) for BIO span tagging."
        )
        text_input = gr.Textbox(label="Persian text", lines=3, rtl=True, placeholder="یک جمله فارسی بنویسید...")
        submit_btn = gr.Button("Tag", variant="primary")
        highlighted_output = gr.HighlightedText(label="Extracted clauses", color_map=SPAN_COLORS, rtl=True)

        gr.Examples(examples=EXAMPLES, inputs=text_input)

        submit_btn.click(fn=tag_text, inputs=text_input, outputs=highlighted_output)
        text_input.submit(fn=tag_text, inputs=text_input, outputs=highlighted_output)

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
