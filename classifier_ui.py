"""Gradio UI for the Persian conditional-type classifier trained by train_classifier.py.

Enter Persian text and it's classified as one of:
  negative  -- بدون شرط (no conditional)
  marked    -- شرطی با نشانه (اگر...) (conditional with an explicit marker word)
  unmarked  -- شرطی بدون نشانه (conditional expressed implicitly)

The predicted class is shown by highlighting the whole input text with that class's color
(gr.HighlightedText), alongside a probability bar per class (gr.Label). Browsers apply the
Unicode bidi algorithm natively, so unlike the matplotlib plots elsewhere in this project, no
reshaping/reordering workaround is needed here for the Persian text to render correctly.

Usage:
    python classifier_ui.py
    python classifier_ui.py --adapter-dir checkpoints/parsbert-lora-conditional-classifier/final

Requirements:
    pip install torch transformers peft gradio
"""

import argparse
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from classifier_common import CLASS_COLORS, CLASS_DISPLAY_FA, CLASS_NAMES, ID2LABEL, LABEL2ID, MAX_LENGTH, MODEL_NAME

EXAMPLES = [
    "اگر باران ببارد، فردا به پارک نمی‌رویم.",
    "بخواب، حالت بهتر می‌شه.",
    "امروز برای شام یه املت خوشمزه درست کردم.",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--adapter-dir",
        default="checkpoints/parsbert-lora-conditional-classifier/final",
        help="Directory saved by train_classifier.py (base model + PEFT adapter + tokenizer).",
    )
    parser.add_argument("--base-model", default=MODEL_NAME)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    return parser.parse_args()


def load_model(adapter_dir, base_model_name):
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"{adapter_dir} not found. Run train_classifier.py first (it saves the adapter there)."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=len(CLASS_NAMES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    return tokenizer, model


def main():
    args = parse_args()
    tokenizer, model = load_model(args.adapter_dir, args.base_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    color_map = {CLASS_DISPLAY_FA[name]: CLASS_COLORS[name] for name in CLASS_NAMES}

    @torch.no_grad()
    def classify(text):
        text = (text or "").strip()
        if not text:
            return [("", None)], {}

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_id = int(probs.argmax())
        pred_display = CLASS_DISPLAY_FA[CLASS_NAMES[pred_id]]

        highlighted = [(text, pred_display)]
        label_scores = {CLASS_DISPLAY_FA[name]: float(probs[i]) for i, name in enumerate(CLASS_NAMES)}
        return highlighted, label_scores

    with gr.Blocks(title="Persian Conditional Classifier") as demo:
        gr.Markdown(
            "# Persian Conditional Classifier\n"
            "Enter a Persian sentence. It is classified as **negative** (no conditional), "
            "**marked** (uses اگر/هرگاه/... ), or **unmarked** (implicit conditional) — a LoRA "
            "adapter fine-tuned on ParsBERT (`" + args.base_model + "`)."
        )
        text_input = gr.Textbox(label="Persian text", lines=3, rtl=True, placeholder="یک جمله فارسی بنویسید...")
        submit_btn = gr.Button("Classify", variant="primary")
        highlighted_output = gr.HighlightedText(label="Classification", color_map=color_map)
        label_output = gr.Label(label="Class probabilities", num_top_classes=len(CLASS_NAMES))

        gr.Examples(examples=EXAMPLES, inputs=text_input)

        submit_btn.click(fn=classify, inputs=text_input, outputs=[highlighted_output, label_output])
        text_input.submit(fn=classify, inputs=text_input, outputs=[highlighted_output, label_output])

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
