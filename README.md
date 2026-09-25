# Persian Conditionals — UI

Gradio UI for the trained LoRA/ParsBERT span tagger (`span_classifier_ui.py`): tags a sentence's
IF (condition) and THEN (consequence) clauses.

Requires a trained adapter checkpoint (produced by `train_span_classifier.py`) under
`checkpoints/`, which this repo already includes under `final/`.

## Setup

```
pip install -r requirements.txt
```

## Run

```
python span_classifier_ui.py
```

Starts a local Gradio server (prints a `http://127.0.0.1:7860` URL to open in a browser).
Use `--adapter-dir` to point at a different checkpoint, and `--share` to create a public link.
