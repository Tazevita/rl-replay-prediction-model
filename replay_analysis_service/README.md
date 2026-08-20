# Replay Analysis Service

This package owns the complete runtime implementation used by replay-viewer:

- Replay reconstruction and feature generation
- GRU checkpoint architecture and batched inference
- Tactical family comparison and goal findings
- Player prediction serialization
- The versioned JSON output contract

Run it from the model repository root:

```sh
prediction-model/.venv/bin/python -m replay_analysis_service \
  path/to/replay.json --all-teams --prediction-interval 1
```

Raw `.replay` files are also accepted when `rrrocket` is present at the model
repository root. Runtime model weights remain data assets under
`prediction-model/artifacts/`; they are not duplicated into the source package.

`replay-goal-runner/run_goal_analysis.py` and `preprocess_replays.py` are thin
compatibility entry points for existing commands and imports.
