"""Fine-tune a LoRA (PEFT) adapter on ParsBERT for Persian conditional-type classification.

Three classes (see classifier_common.py):
  negative  -- no conditional relation
  marked    -- conditional with an explicit marker word (اگر, هرگاه, ...)
  unmarked  -- conditional expressed implicitly, without a marker word

Labels are read from each example's `prompt_variant` field, so no extra annotation is needed --
just the two generated datasets:
  - persian_conditionals_bio.jsonl  (marked / unmarked)
  - persian_negatives_bio.jsonl     (negative)

Usage:
    python train_classifier.py
    python train_classifier.py --epochs 15 --batch-size 32 --output-dir checkpoints/run2

Requirements (on top of what generate_conditionals.ipynb already needs):
    pip install torch transformers peft datasets scikit-learn pandas matplotlib
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this is a CLI training script, not a notebook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from classifier_common import CLASS_NAMES, ID2LABEL, LABEL2ID, MAX_LENGTH, MODEL_NAME, load_labeled_texts


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--positives-path", default="persian_conditionals_bio.jsonl")
    parser.add_argument("--negatives-path", default="persian_negatives_bio.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/parsbert-lora-conditional-classifier")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--eval-save-steps", type=int, default=50, help="checkpoint + eval every N steps")
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require_data_file(path, min_examples, hint):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. {hint}")
    n = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if n < min_examples:
        raise ValueError(f"{path} only has {n} example(s) (need >= {min_examples}). {hint}")
    return path


def build_split_datasets(positives_path, negatives_path, seed):
    labeled = load_labeled_texts(positives_path, negatives_path)
    texts = [r["text"] for r in labeled]
    labels = [r["label"] for r in labeled]

    # 60 / 20 / 20 train / val / test, stratified by label.
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.4, random_state=seed, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=seed, stratify=temp_labels
    )

    def to_dataset(t, l):
        return Dataset.from_dict({"text": t, "label": l})

    splits = DatasetDict(
        train=to_dataset(train_texts, train_labels),
        validation=to_dataset(val_texts, val_labels),
        test=to_dataset(test_texts, test_labels),
    )
    print("Split sizes:")
    for name, ds in splits.items():
        counts = {CLASS_NAMES[i]: ds["label"].count(i) for i in range(len(CLASS_NAMES))}
        print(f"  {name:<10} n={len(ds):<5} {counts}")
    return splits


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    accuracy = float((preds == labels).mean())
    return {"accuracy": accuracy, "macro_f1": f1, "macro_precision": precision, "macro_recall": recall}


def save_confusion_matrix(cm, output_path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")  # single hue, light->dark: sequential encoding for counts
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test split)")

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    positives_path = require_data_file(
        args.positives_path,
        min_examples=10,
        hint="Run generate_conditionals.ipynb first.",
    )
    negatives_path = require_data_file(
        args.negatives_path,
        min_examples=10,
        hint="Run generate_negatives.ipynb (with a real CONFIG['num_examples'], not the 2-example "
        "smoke test) first.",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = build_split_datasets(positives_path, negatives_path, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized = splits.map(tokenize_fn, batched=True, remove_columns=["text"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(CLASS_NAMES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query", "value"],  # BERT-style self-attention projections
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        eval_strategy="steps",
        eval_steps=args.eval_save_steps,
        save_strategy="steps",
        save_steps=args.eval_save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=25,
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()

    # ---- final report on the held-out test split ----
    test_output = trainer.predict(tokenized["test"])
    logits = test_output.predictions
    true_ids = test_output.label_ids
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred_ids = np.argmax(logits, axis=-1)

    report = classification_report(
        true_ids, pred_ids, target_names=CLASS_NAMES, digits=3, zero_division=0
    )
    print("\nTest classification report:\n")
    print(report)
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(true_ids, pred_ids, labels=range(len(CLASS_NAMES)))
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    save_confusion_matrix(cm, output_dir / "confusion_matrix.png")

    test_texts = tokenized["test"]["text"] if "text" in tokenized["test"].column_names else None
    if test_texts is None:
        # text column was removed by tokenize_fn's remove_columns; re-pull from the pre-tokenized split
        test_texts = splits["test"]["text"]

    results_df = pd.DataFrame(
        {
            "text": test_texts,
            "true_label": [CLASS_NAMES[i] for i in true_ids],
            "predicted_label": [CLASS_NAMES[i] for i in pred_ids],
            "correct": true_ids == pred_ids,
            **{f"prob_{name}": probs[:, i] for i, name in enumerate(CLASS_NAMES)},
        }
    )
    csv_path = output_dir / "test_predictions.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")  # BOM: renders correctly in Excel
    print(f"\nSaved {len(results_df)} test predictions to {csv_path}")

    summary = {
        "accuracy": float((true_ids == pred_ids).mean()),
        "n_test": len(true_ids),
        "class_names": CLASS_NAMES,
    }
    (output_dir / "test_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    (final_dir / "label_map.json").write_text(
        json.dumps({"id2label": ID2LABEL, "label2id": LABEL2ID}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved final PEFT adapter + tokenizer to {final_dir}")


if __name__ == "__main__":
    main()
