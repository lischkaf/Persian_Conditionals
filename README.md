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

## Limitations
- Only works reliably if IF and THEN clause are separated by a comma. -> throw out comma in preprocessing (training and inference)
- Result on test split has to be taken with a grain of salt (no deduplication, no held-out clusters/topics)
- Only works reliably on single sentences. (easily fixed at inference)
- Sentences w/o conditional and with comma work sometimes (223/1000 negative examples have a comma).
- unusual grammatical constructions (verb position!) don't work reliably
- conditional markers not in the training set don't work when the conditional is the second part of the sentence
- conditional sentence as the second part of the sentence doesn't work
- many subordinary clauses (e.g. temporal, causal) are tagged as conditionals

## Further work
- separate sentences for inference
- train again with commas removed (and adapt inference)
- generate new training set:
  - save all the choices made for sample (e.g. topic, register) as annotation
  - more non-conditionals with subordinate clause
  - explicitly ask for unusual grammatical constructions (e.g. verb at the beginning, subordinary clause as the second part of a sentence)
  - give a full list of Persian conditional markers and make sure all are used
- new training run with held-out topics
- look even harder at attention scores to explain decisions
