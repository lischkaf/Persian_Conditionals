"""Fine-tune a LoRA (PEFT) adapter on ParsBERT for IF/THEN span tagging (token classification).

Unlike train_classifier.py (which classifies a whole sentence as negative/marked/unmarked), this
predicts a BIO tag per token -- which words are the condition (IF) clause and which are the
consequence (THEN) clause -- directly from the `tokens`/`tags` fields already in the dataset:

    O   B-IF  I-IF  B-THEN  I-THEN

Trained on both:
  - persian_conditionals_bio.jsonl  (real IF/THEN spans)
  - persian_negatives_bio.jsonl     (every token tagged O -- teaches the model not to over-tag)

Usage:
    python train_span_classifier.py
    python train_span_classifier.py --epochs 15 --batch-size 32 --output-dir checkpoints/run2

Reports two kinds of metrics on the test split:
  - token-level (classification_report.txt / confusion_matrix.png / test_predictions.csv):
    per-token tag accuracy -- lenient, a span with one wrong boundary token still scores mostly
    right.
  - entity-level / exact-match (entity_classification_report.txt / test_predictions_sentence
    _level.csv): via seqeval, a predicted IF/THEN span only counts as correct if its full token
    range exactly matches the gold span (right type *and* right boundaries) -- this is "how often
    are the spans fully correct", plus a whole-sentence exact-tag-sequence-match rate.

Requirements (on top of what generate_conditionals.ipynb already needs):
    pip install torch transformers peft datasets scikit-learn pandas matplotlib seqeval
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
from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.metrics import f1_score as seqeval_f1_score
from seqeval.metrics import precision_score as seqeval_precision_score
from seqeval.metrics import recall_score as seqeval_recall_score
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from span_classifier_common import (
    BIO_ID2LABEL,
    BIO_LABEL2ID,
    BIO_TAGS,
    MAX_LENGTH,
    MODEL_NAME,
    load_token_tag_examples,
    tags_to_clauses,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--positives-path", default="persian_conditionals_bio.jsonl")
    parser.add_argument("--negatives-path", default="persian_negatives_bio.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/parsbert-lora-span-classifier")
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
    records = load_token_tag_examples([positives_path, negatives_path])
    strata = [r["prompt_variant"] for r in records]  # stratify only, not a model input/target

    # 60 / 20 / 20 train / val / test, stratified by prompt_variant (marked / unmarked / negative)
    # so each split has representative coverage of all three example types.
    train_recs, temp_recs, train_strata, temp_strata = train_test_split(
        records, strata, test_size=0.4, random_state=seed, stratify=strata
    )
    val_recs, test_recs, _, _ = train_test_split(
        temp_recs, temp_strata, test_size=0.5, random_state=seed, stratify=temp_strata
    )

    def to_dataset(recs):
        return Dataset.from_dict({"tokens": [r["tokens"] for r in recs], "tags": [r["tags"] for r in recs]})

    splits = DatasetDict(train=to_dataset(train_recs), validation=to_dataset(val_recs), test=to_dataset(test_recs))
    print("Split sizes:")
    for name, recs in [("train", train_recs), ("validation", val_recs), ("test", test_recs)]:
        counts = {}
        for r in recs:
            counts[r["prompt_variant"]] = counts.get(r["prompt_variant"], 0) + 1
        print(f"  {name:<10} n={len(recs):<5} {counts}")
    return splits


def make_tokenize_and_align_fn(tokenizer, max_length):
    def tokenize_and_align_labels(batch):
        tokenized = tokenizer(
            batch["tokens"], truncation=True, max_length=max_length, is_split_into_words=True
        )
        all_labels = []
        for i, tags in enumerate(batch["tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # first subword of this word: real label
                    label_ids.append(BIO_LABEL2ID[tags[word_idx]])
                else:
                    # continuation subword of the same word: ignored in the loss
                    label_ids.append(-100)
                previous_word_idx = word_idx
            all_labels.append(label_ids)
        tokenized["labels"] = all_labels
        return tokenized

    return tokenize_and_align_labels


def compute_metrics(eval_pred):
    """Token-level macro P/R/F1 (used for early stopping / best-checkpoint selection, unchanged),
    plus entity-level (exact span match) and whole-sentence exact-match metrics via seqeval, so
    training progress on "are the spans fully correct" is visible in the eval logs too."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    flat_preds, flat_labels = [], []
    seq_true_tags, seq_pred_tags = [], []
    n_exact_sentence_matches = 0
    for pred_seq, label_seq in zip(preds, labels):
        true_tags, pred_tags = [], []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            flat_preds.append(p)
            flat_labels.append(l)
            true_tags.append(BIO_ID2LABEL[l])
            pred_tags.append(BIO_ID2LABEL[p])
        seq_true_tags.append(true_tags)
        seq_pred_tags.append(pred_tags)
        if true_tags == pred_tags:
            n_exact_sentence_matches += 1

    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_labels, flat_preds, average="macro", zero_division=0, labels=list(range(len(BIO_TAGS)))
    )
    token_accuracy = float(np.mean(np.array(flat_preds) == np.array(flat_labels))) if flat_labels else 0.0

    return {
        "token_accuracy": token_accuracy,
        "macro_f1": f1,  # token-level: kept as the model-selection metric (unchanged behavior)
        "macro_precision": precision,
        "macro_recall": recall,
        "entity_f1": seqeval_f1_score(seq_true_tags, seq_pred_tags),
        "entity_precision": seqeval_precision_score(seq_true_tags, seq_pred_tags),
        "entity_recall": seqeval_recall_score(seq_true_tags, seq_pred_tags),
        "sentence_exact_match": n_exact_sentence_matches / len(preds) if len(preds) else 0.0,
    }


def save_confusion_matrix(cm, output_path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")  # single hue, light->dark: sequential encoding for counts
    ax.set_xticks(range(len(BIO_TAGS)))
    ax.set_yticks(range(len(BIO_TAGS)))
    ax.set_xticklabels(BIO_TAGS, rotation=30, ha="right")
    ax.set_yticklabels(BIO_TAGS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test split, token-level)")

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    positives_path = require_data_file(
        args.positives_path, min_examples=10, hint="Run generate_conditionals.ipynb first."
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
    if not tokenizer.is_fast:
        raise RuntimeError(
            f"{args.model_name} did not load a fast (Rust) tokenizer, which word-level label "
            "alignment (tokenized.word_ids(...)) requires."
        )

    tokenize_and_align_labels = make_tokenize_and_align_fn(tokenizer, args.max_length)
    tokenized = splits.map(tokenize_and_align_labels, batched=True, remove_columns=["tokens", "tags"])
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    base_model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(BIO_TAGS),
        id2label=BIO_ID2LABEL,
        label2id=BIO_LABEL2ID,
    )
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
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

    # ---- final report on the held-out test split (token-level, ignoring subword continuations) ----
    test_output = trainer.predict(tokenized["test"])
    logits = test_output.predictions
    labels = test_output.label_ids
    pred_ids = np.argmax(logits, axis=-1)

    rows = []
    sentence_rows = []
    flat_true, flat_pred = [], []
    seq_true_tags, seq_pred_tags = [], []
    n_exact_sentence_matches = 0
    test_tokens = splits["test"]["tokens"]
    for sent_idx in range(len(test_tokens)):
        # Recompute word_ids for this sentence (not kept by the earlier batched tokenize/align
        # step) to map each subword prediction back to its source token.
        encoded = tokenizer(test_tokens[sent_idx], truncation=True, max_length=args.max_length, is_split_into_words=True)
        seen_words = set()
        true_tags, pred_tags = [], []
        for subword_idx, word_idx in enumerate(encoded.word_ids()):
            if word_idx is None or word_idx in seen_words or labels[sent_idx][subword_idx] == -100:
                continue
            seen_words.add(word_idx)
            true_id = int(labels[sent_idx][subword_idx])
            pred_id = int(pred_ids[sent_idx][subword_idx])
            flat_true.append(true_id)
            flat_pred.append(pred_id)
            true_tags.append(BIO_ID2LABEL[true_id])
            pred_tags.append(BIO_ID2LABEL[pred_id])
            rows.append(
                {
                    "sentence_id": sent_idx,
                    "token": test_tokens[sent_idx][word_idx],
                    "true_tag": BIO_ID2LABEL[true_id],
                    "predicted_tag": BIO_ID2LABEL[pred_id],
                    "correct": true_id == pred_id,
                }
            )

        seq_true_tags.append(true_tags)
        seq_pred_tags.append(pred_tags)
        exact_match = true_tags == pred_tags
        n_exact_sentence_matches += exact_match

        # true_tags/pred_tags only cover words that survived truncation to max_length, in order.
        tokens_for_sentence = test_tokens[sent_idx][: len(true_tags)]
        true_clauses = tags_to_clauses(tokens_for_sentence, true_tags)
        pred_clauses = tags_to_clauses(tokens_for_sentence, pred_tags)
        sentence_rows.append(
            {
                "sentence_id": sent_idx,
                "text": " ".join(tokens_for_sentence),
                "true_spans": json.dumps(true_clauses, ensure_ascii=False),
                "predicted_spans": json.dumps(pred_clauses, ensure_ascii=False),
                "spans_exact_match": exact_match,
            }
        )

    # ---- token-level report (lenient: scores each token independently) ----
    report = classification_report(
        flat_true, flat_pred, labels=list(range(len(BIO_TAGS))), target_names=BIO_TAGS, digits=3, zero_division=0
    )
    print("\nTest classification report (token-level):\n")
    print(report)
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(flat_true, flat_pred, labels=list(range(len(BIO_TAGS))))
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    save_confusion_matrix(cm, output_dir / "confusion_matrix.png")

    results_df = pd.DataFrame(rows)
    csv_path = output_dir / "test_predictions.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")  # BOM: renders correctly in Excel
    print(f"\nSaved {len(results_df)} token-level test predictions to {csv_path}")

    # ---- entity-level / exact-match report: "how often are the spans fully correct" ----
    entity_report = seqeval_classification_report(seq_true_tags, seq_pred_tags, digits=3, zero_division=0)
    sentence_exact_match_rate = n_exact_sentence_matches / len(test_tokens) if test_tokens else 0.0
    print("\nEntity-level (exact span match) report:\n")
    print(entity_report)
    print(f"Whole-sentence exact tag-sequence match: {sentence_exact_match_rate:.3f}")
    (output_dir / "entity_classification_report.txt").write_text(
        entity_report + f"\nwhole-sentence exact match: {sentence_exact_match_rate:.3f}\n", encoding="utf-8"
    )

    sentence_df = pd.DataFrame(sentence_rows)
    sentence_csv_path = output_dir / "test_predictions_sentence_level.csv"
    sentence_df.to_csv(sentence_csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(sentence_df)} sentence-level (exact-match) test predictions to {sentence_csv_path}")

    summary = {
        "token_accuracy": float(np.mean(np.array(flat_true) == np.array(flat_pred))) if flat_true else 0.0,
        "entity_f1": seqeval_f1_score(seq_true_tags, seq_pred_tags),
        "entity_precision": seqeval_precision_score(seq_true_tags, seq_pred_tags),
        "entity_recall": seqeval_recall_score(seq_true_tags, seq_pred_tags),
        "sentence_exact_match": sentence_exact_match_rate,
        "n_test_sentences": len(test_tokens),
        "n_test_tokens": len(flat_true),
        "tag_names": BIO_TAGS,
    }
    (output_dir / "test_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    (final_dir / "label_map.json").write_text(
        json.dumps({"id2label": BIO_ID2LABEL, "label2id": BIO_LABEL2ID}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved final PEFT adapter + tokenizer to {final_dir}")


if __name__ == "__main__":
    main()
