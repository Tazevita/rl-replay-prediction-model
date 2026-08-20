# Intent prediction model

A memory-bounded PyTorch GRU that predicts one exclusive player intent plus the
player's endpoint position and facing direction from the previous second of
normalized Rocket League game state. It consumes the SQLite replay shards
produced by the preprocessing pipeline in the parent directory.

The current intent classes are `CHALLENGE`, `POSSESS`, `SUPPORT`, `SHADOW`,
`ROTATE`, `HOLD`, `BOOST_DETOUR`, `BUMP`, `PRESSURE`, `REPOSITION`, `DEFEND`,
and `OTHER`.

## Chosen design

- Input: the existing 11 states sampled at 10 Hz, spanning one second.
- Model: one 96-unit GRU followed by a 64-unit classification head.
- Objective: confidence-weighted cross entropy for intent, Huber loss for
  endpoint position, and cosine loss for endpoint facing direction.
- Imbalance: bounded inverse-square-root class weights.
- Scaling: per-feature mean and standard deviation from training replays, stored
  inside the checkpoint and applied automatically during inference.
- Validation: split by whole replay, never by overlapping sequence window.
- Selection: best combined validation loss with early stopping, so endpoint and
  intent quality both participate in checkpoint selection.
- Memory: SQLite tracks are loaded lazily into an LRU capped at four tracks.
  Track-aware batches keep all windows from a track adjacent, avoiding repeated
  multi-megabyte SQLite reads even with a small cache.
- Inference: a rolling 11-state buffer; no transformer-style cache is needed.

These defaults are intentionally small. Increase model size only after the
validation results show underfitting.

## Batch size

`--batch-size 128` means 128 one-second player windows per optimizer update. It
does not mean 128 replay files. A batch may contain windows from multiple replay
files. The whole-replay distinction matters for train/validation splitting, not
for gradient batch construction.

Larger batches use more activation memory. If memory is tight, try 64 or 32.
Changing batch size does not change the length of history seen by the model.

## Install

The system `python3` in this workspace is Python 3.14 and does not currently
have a compatible PyTorch installation. Use Python 3.12 or 3.13 for the model
environment. On this Apple Silicon machine, one option is:

```sh
brew install python@3.12
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

PyTorch automatically exposes the Apple `mps` device when the installed build
and macOS version support it.

## Train

From `prediction-model/`:

```sh
.venv/bin/python -m prediction_model.train \
  --manifest ../training_output/manifest.json \
  --output artifacts/intent_gru.pt
```

Useful memory controls:

```sh
.venv/bin/python -m prediction_model.train \
  --manifest ../training_output/manifest.json \
  --output artifacts/intent_gru.pt \
  --batch-size 64 \
  --cache-tracks 2 \
  --workers 0
```

Keep `--workers 0` when memory is the priority. Each additional worker has its
own SQLite connection and track cache. The training command prints the replay
split and label counts, saves the best checkpoint, and writes per-epoch metrics
to `artifacts/intent_gru.history.json`.

The current dataset contains only three replays. The automatic split therefore
uses two for training and one for validation. This is suitable for proving the
pipeline works, but more independent replays are needed before treating the
validation score as a reliable estimate of live performance.

## Predict

The inference input is a JSON array containing exactly 11 normalized feature
vectors in the same order as `training_output/manifest.json`:

```sh
.venv/bin/python -m prediction_model.predict \
  artifacts/intent_gru.pt \
  path/to/window.json
```

The result is a JSON object containing all intent probabilities, the target
time, endpoint position in canonical team-relative Rocket League units, and a
unit forward vector. With the recommended checkpoints, endpoints are predicted
at `+1.0s`, `+2.0s`, and `+3.5s`.
For a live process, instantiate `RollingIntentPredictor` and call `add_state`
at 10 Hz. It returns `None` until the first complete one-second window exists:

```python
from prediction_model.predict import RollingIntentPredictor

predictor = RollingIntentPredictor("artifacts/intent_gru.pt")
probabilities = predictor.add_state(normalized_feature_vector)
if probabilities is not None:
    predicted_intent = max(probabilities, key=probabilities.get)
```

Use `add_state_prediction` or `predict_endpoint_window` when position and facing
are needed as well as intent:

```python
prediction = predictor.add_state_prediction(normalized_feature_vector)
if prediction is not None:
    print(prediction.probabilities)
    print(prediction.position, prediction.forward, prediction.seconds)
```

Checkpoint version 2 requires format-6 replay datasets. Re-run the three parent
pipeline commands and retrain all checkpoints after upgrading; version-1 intent
checkpoints do not contain endpoint heads.

Call `reset()` at kickoff, replay reset, or whenever state continuity is lost.

## Test

```sh
.venv/bin/python -m pytest
```

## Decisions to revisit with more data

- Label quality: intents are currently future-trajectory heuristics, not human
  annotations. Review false positives before tuning the neural network.
- Replay split: use separate train, validation, and test replay groups once at
  least several dozen independent replays are available.
- Calibration: add temperature scaling on a dedicated calibration split if the
  numeric probabilities will drive thresholds rather than just ranking intents.
