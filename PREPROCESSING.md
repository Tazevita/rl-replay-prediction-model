# Replay preprocessing

## One-command pipeline

The default folder flow is:

```text
replay_input/       Raw .replay files
rrrocket_parsed/    Network-parsed rrrocket JSON
training_output/    Custom transformed dataset and metadata
```

Place replay files in `replay_input/`, then run:

```sh
python3 run_pipeline.py
```

The pipeline produces one transformed shard per replay plus a global manifest:

```text
training_output/
  manifest.json
  replays/
    example.sqlite
    example.metadata.json
```

`manifest.json` lists every shard, the shared feature schema, and aggregate
intent, mechanic, and event counts. Replays whose parsed JSON is newer than the
source replay are reused. Transformed shards are also reused unless their
parsed replay or the transform code changed. Pass `--force` to rebuild both
stages. Nested directories under `replay_input/` are preserved in
`rrrocket_parsed/` and `training_output/replays/`.

Use `python3 run_pipeline.py --help` to override the three directories,
manifest filename, or expected team size.

Build datasets for non-overlapping tactical prediction windows with separate
output directories:

```sh
python3 run_pipeline.py --output-dir training_output_0_1 \
  --target-offset-seconds 0 --horizon-seconds 1
python3 run_pipeline.py --output-dir training_output_1_2 \
  --target-offset-seconds 1 --horizon-seconds 1
python3 run_pipeline.py --output-dir training_output_2_3_5 \
  --target-offset-seconds 2 --horizon-seconds 1.5
```

Each model still receives only the history ending at the current timestamp.
The target offset changes which future interval supplies its training label; it
does not add future states to model input.

## Custom transform

`preprocess_replays.py` converts rrrocket network JSON into fixed-shape,
ego-centric action-classification examples.

```sh
python3 preprocess_replays.py output.json -o training.sqlite
```

Directories are accepted and searched recursively for JSON files:

```sh
python3 preprocess_replays.py replays/ -o doubles-training.sqlite
```

The output is a SQLite shard containing float32 feature tracks and compact
indexed examples. Each track stores every ego-centric state once. An example
stores its track and ending-state index rather than another copy of its history.
Each example also stores the ego player's normalized canonical position and unit
facing vector at the exact end of its target window. The three recommended
datasets therefore target player state at `+1.0s`, `+2.0s`, and `+3.5s`.
Use `ReplayDataset` to assemble the configured history window:

```python
from training_dataset import ReplayDataset

with ReplayDataset("training.sqlite") as dataset:
    example = dataset[0]
    assert example.shape == (dataset.sequence_length, dataset.feature_count)
    features = example.features  # contiguous float32 memoryview
```

`features` is a flat zero-copy view with logical shape
`[sequence_length][feature_count]`. The adjacent metadata file lists every
feature in column order, normalization constants, label thresholds, and
per-replay label counts. The same metadata is embedded in the SQLite shard.

Player slots are assigned once from the replay-wide roster and never compacted.
If a player is demolished or otherwise temporarily absent, their original slot
contains zeroes with its `valid` feature set to zero. Other players do not move
between teammate or opponent slots. Replays with more unique players on a team
than `--team-size` are rejected because they cannot be represented without
reusing a slot.

## Labels

Each example has an exclusive intent label, an exclusive mechanic label, and
zero or more event labels. They are initial heuristics inferred from the
player's current state and future trajectory.

Intent labels:

- `CHALLENGE`: the player moves and closes enough distance to come within the
  configured distance of the ball during the prediction horizon. It also
  recognizes a sustained, accelerating, ball-facing approach that remains
  outside contact range at the end of the horizon.
- `POSSESS`: the player moves with the ball inside a persistent control envelope.
  Car and ball velocity are compared throughout the trajectory, with a wider
  tolerance while airborne and one brief interruption allowed for a touch
  impulse. Proximity without coupled motion is not possession.
- `SUPPORT`: the player maintains useful goal-side spacing behind a teammate
  who remains engaged as first man. The relationship is measured across the
  trajectory, allows lateral and controlled-retreat coverage, and expands for
  teammates making aerial plays.
- `CHERRY_PICK`: the player stays in the attacking half, persistently ahead of
  both the ball and an engaged teammate, as an upfield passing option. The
  player must remain outside challenge range and cannot be rotating back or
  taking a qualifying large-boost detour.
- `SHADOW`: the player retreats in the defensive lane from an incoming ball
  controlled by an opponent while staying outside challenge range. The lane is
  projected from ball speed and side/back-wall bounces, and excludes cars
  already inside their own net.
- `CLOSE_ROTATE`: the player returns toward their own goal on the same side as
  the ball. Wide returns can qualify through accumulated path and own-goal
  progress even when net Y backtracking is small.
- `FAR_ROTATE`: the same defensive return on the opposite side from the ball,
  without approaching challenge distance.
- `HOLD`: the player has little displacement over the prediction horizon.
- `BOOST_DETOUR`: the player leaves the useful path to the play to route toward
  a large boost pad. Rotations, bumps, and pressure are classified first; boost
  detours take precedence over cherry-picking, support, and attack. Incidental
  pickups remain `BOOST_PICKUP` events.
- `BUMP`: the player reaches collision-scale proximity while strongly facing
  and driving into an opponent. Ego must contribute more closing speed than the
  opponent. Near the ball, opponent alignment and approach must be substantially
  stronger than ball alignment and approach; ambiguous 50/50s remain challenges.
- `PRESSURE`: the player closes on an opponent controlling the ball without
  entering challenge range.
- `ATTACK`: the player pursues a loose ball advancing toward the opponent's end.
  The car remains behind and aligned with the ball without falling materially
  farther away; possession, challenges, pressure, and teammate support take
  precedence.
- `REPOSITION`: the player makes a substantial lateral adjustment that does not
  match another tactical intent. Lateral evidence is accumulated across the
  complete path rather than inferred from only its endpoints.
- `DEFEND`: the player holds their own goal area, enters the goal, or moves
  substantially closer to the lane between an own-half ball and the goal.
- `OTHER`: neither rule applies.

Mechanic labels:

- `GROUNDED`: the car is upright and below the configured airborne height.
- `AERIAL`: the car is above the airborne height and remains controlled.
- `RECOVERING`: the car is tilted substantially or descending toward a landing.

Event labels:

- `SHOT`: a close ball interaction sends the ball toward the opponent's end.
- `CLEAR`: the same interaction starts deep in the player's own half.
- `BOOST_PICKUP`: boost rises by the configured amount during the horizon.
- `DEMOLITION`: a nearby opponent disappears from active car states.

Examples whose history or future-label window crosses a goal are excluded to
avoid labeling replay resets as actions. These heuristics cannot reliably
identify every fake challenge or ambiguous defensive movement, and event labels infer
outcomes from physics rather than authoritative touch records. Review a sample
manually and tune the thresholds before treating them as ground truth.

Only replay states marked `Active` are included. Countdown and post-goal replay
states are excluded from both the input history and future labeling window.

Demolished cars are treated as absent for the standard three-second respawn
window. The extractor also rejects bots, non-Soccar modes, and replays whose
team size differs from `--team-size`, preventing incompatible examples from
being mixed silently.

Run `python3 preprocess_replays.py --help` for threshold, history, sampling,
and horizon options. Keep the raw rrrocket JSON so datasets can be regenerated
when label definitions change.

## Action timeline

Print the current action for every player once per second:

```sh
python3 describe_actions.py training.sqlite --replay-json output.json
```

The raw replay JSON is optional, but allows the reader to replace anonymized
player IDs with names. Useful display options include:

```sh
# Only print action transitions.
python3 describe_actions.py training.sqlite --replay-json output.json --changes-only

# Replay reports with real-time delays at 4x speed.
python3 describe_actions.py training.sqlite --replay-json output.json --realtime --speed 4

# Inspect a short elapsed-time range with confidence values.
python3 describe_actions.py training.sqlite --replay-json output.json \
  --start 50 --end 80 --interval 0.5 --show-confidence
```

The timeline describes preprocessing heuristics. Replace its label source with
model outputs later if you want to narrate actual predictions.
