# Persian Conditionals — IF/THEN Span Tagger

A LoRA adapter fine-tuned on ParsBERT (`HooshvareLab/bert-fa-zwnj-base`) that tags each token of a
Persian sentence as part of the **IF** clause (protasis/condition), the **THEN** clause
(apodosis/consequence), or neither (`O`) — standard BIO sequence labeling
(`O / B-IF / I-IF / B-THEN / I-THEN`). Includes a Gradio UI (`span_classifier_ui.py`) for
interactive tagging.

This README covers only the span-tagging pipeline (data generation → training → UI). The repo
also contains a separate sentence-level classifier (`classifier_common.py` / `train_classifier.py`
/ `classifier_ui.py`, negative/marked/unmarked) not described here.

## Setup

```
pip install -r requirements.txt
```

## Pipeline

### 1. Generate training data (`generate_conditionals.ipynb`, `generate_negatives.ipynb`)

Both notebooks call an OpenAI-compatible chat model to generate short Persian sentences and write
them to JSONL in a shared schema: one example per line with `id`, `text`, `tokens`, `tags`
(BIO), `spans`, and `prompt_variant`. Requires `pip install openai` and an `OPENAI_API_KEY`
(optionally via a `.env` file).

- **`generate_conditionals.ipynb` → `persian_conditionals_bio.jsonl`** (1000 examples, 500/500
  split): alternates between two prompts —
  - `marked`: the sentence must use an explicit conditional marker (`اگر`, `هرگاه`,
    `در صورتی که`, `چنانچه`, `به شرطی که`, ...).
  - `unmarked`: the sentence must express a conditional relation *without* a marker word (e.g.
    imperative + result juxtaposition, or subjunctive/future mood implying a hypothetical).

  The model returns the sentence plus its IF/THEN clauses as exact verbatim substrings (via a
  structured JSON schema response), which are located by character offset, tokenized with a
  regex tokenizer (`[\w‌]+` for words, one token per punctuation character), and converted to
  per-token BIO tags.
- **`generate_negatives.ipynb` → `persian_negatives_bio.jsonl`** (1000 examples): same schema and
  topic/register sampling, but the system prompt explicitly forbids any conditional marker or
  implicit conditional construction (mirroring what the `unmarked` prompt is asked to produce).
  Every token is tagged `O`; `spans` is always empty; `prompt_variant` is `"negative"`. Generated
  text is additionally checked against a list of conditional marker words as a cheap sanity filter.

Both notebooks sample from the same 15 topics (economy, weather, travel, education, family,
technology, health, sports, environment, work, food, traffic, friendship, politics, music/art)
and two registers (colloquial/formal), so positive and negative examples cover comparable ground.

### 2. Train (`train_span_classifier.py`)

```
python train_span_classifier.py
python train_span_classifier.py --epochs 15 --batch-size 32 --output-dir checkpoints/run2
```

- Loads both JSONL files, does a seeded 60/20/20 train/val/test split (`train_test_split`,
  `seed=42`) stratified by `prompt_variant` so marked/unmarked/negative examples are represented
  proportionally in every split.
- Fine-tunes a LoRA adapter (`peft`, `target_modules=["query", "value"]`, rank 8 by default) on
  top of `AutoModelForTokenClassification` for ParsBERT — only the adapter weights are trained,
  the base model stays frozen.
- Subword alignment: only the first subword of each word carries the label during training
  (continuation subwords are masked with `-100`), matching how predictions are read back out at
  inference.
- Reports two kinds of metrics on the held-out test split (current run: 400 sentences,
  5227 tokens):
  - **token-level** (`classification_report.txt`, `confusion_matrix.png`,
    `test_predictions.csv`): lenient, scores each token independently
    (token accuracy ≈ 0.996).
  - **entity-level / exact-span-match** via `seqeval` (`entity_classification_report.txt`,
    `test_predictions_sentence_level.csv`): a predicted span only counts as correct if its full
    token range exactly matches the gold span (entity F1 ≈ 0.963), plus a whole-sentence
    exact-tag-sequence match rate (≈ 0.96).
- Saves the final adapter + tokenizer + label map to `<output-dir>/final/` (this repo ships one at
  `checkpoints/parsbert-lora-span-classifier/final/`).

### 3. Interactive UI (`span_classifier_ui.py`)

```
python span_classifier_ui.py
```

Starts a local Gradio server (prints a `http://127.0.0.1:7860` URL to open in a browser).
Use `--adapter-dir` to point at a different checkpoint, and `--share` to create a public link.

## Limitations
- Only works reliably if IF and THEN clause are separated by a comma. -> throw out comma in preprocessing (training and inference)
- Result on test split has to be taken with a grain of salt (no deduplication, no held-out clusters/topics)
- Only works reliably on single sentences. (easily fixed at inference)
- Sentences w/o conditional and with comma work sometimes (223/1000 negative examples have a comma).
- unusual grammatical constructions (verb position!) don't work reliably
- conditional markers not in the training set don't work (when the conditional is the second part of the sentence)
- conditional sentence as the second part of the sentence doesn't work
- many subordinary clauses (e.g. temporal, causal) are tagged as conditionals

## Further work
- separate sentences for inference
- train again with commas removed (and adapt inference)
- generate new training set:
  - tech debt:
    - save all the choices made for sample (e.g. topic, register) as annotation
    - one python script instead of two notebooks
  - more non-conditionals with subordinate clause
  - explicitly ask for unusual grammatical constructions (e.g. verb at the beginning, subordinary clause as the second part of a sentence)
  - give a full list of Persian conditional markers and make sure all are used
- new training run with held-out topics
- look even harder at attention scores to explain decisions
- compare against LLM-annotated human-written corpora as training data (for more representative language) on hand-annotated eval dataset
