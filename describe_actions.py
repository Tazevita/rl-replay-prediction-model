#!/usr/bin/env python3
"""Print an action timeline from a preprocessed Rocket League dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import time as time_module
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from training_dataset import ReplayDataset


ACTION_WORDS = {
    "CHALLENGE": "challenging",
    "POSSESS": "possessing",
    "SUPPORT": "supporting",
    "SHADOW": "shadowing",
    "CLOSE_ROTATE": "rotating close to the ball",
    "FAR_ROTATE": "rotating opposite the ball",
    "HOLD": "holding",
    "OTHER": "other",
    "BOOST_DETOUR": "detouring for boost",
    "BUMP": "pursuing a bump",
    "PRESSURE": "pressuring",
    "REPOSITION": "repositioning",
    "DEFEND": "defending",
    "ATTACK": "attacking",
    "CHERRY_PICK": "waiting upfield for a pass",
}
MECHANIC_WORDS = {
    "GROUNDED": "grounded",
    "AERIAL": "aerial",
    "RECOVERING": "recovering",
}
EVENT_WORDS = {
    "SHOT": "shot",
    "CLEAR": "clear",
    "BOOST_PICKUP": "boost pickup",
    "DEMOLITION": "demolition",
}
TEAM_NAMES = {0: "Blue", 1: "Orange"}


@dataclass
class PlayerAction:
    player_id: str
    team: int
    label: str
    confidence: float
    mechanic: str = "GROUNDED"
    mechanic_confidence: float = 0.0
    events: tuple[str, ...] = ()


@dataclass
class TimelineSample:
    time: float
    frame: int
    game_seconds: float | None
    overtime: bool
    actions: dict[str, PlayerAction]


@dataclass
class Gap:
    start: float
    end: float


def default_metadata_path(dataset: Path) -> Path:
    name = dataset.name
    for suffix in (".sqlite", ".db"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return dataset.with_name(f"{name}.metadata.json")


def load_metadata(dataset: Path, metadata_path: Path | None) -> dict[str, Any]:
    path = metadata_path or default_metadata_path(dataset)
    if not path.is_file():
        raise FileNotFoundError(f"metadata file not found: {path}")
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def load_timeline(
    dataset: Path, metadata: dict[str, Any]
) -> dict[str, list[TimelineSample]]:
    feature_names = metadata.get("feature_names", [])
    clock_index = feature_names.index("clock_fraction")
    clock_valid_index = feature_names.index("clock_valid")
    overtime_index = feature_names.index("overtime")
    grouped: dict[str, dict[float, TimelineSample]] = {}

    with ReplayDataset(dataset) as source:
        for row_number, row in enumerate(source, 1):
            replay_id = row.replay_id
            sample_time = row.time
            current_features = row.current_features
            game_seconds = (
                float(current_features[clock_index]) * 300.0
                if current_features[clock_valid_index] >= 0.5
                else None
            )
            sample = grouped.setdefault(replay_id, {}).setdefault(
                sample_time,
                TimelineSample(
                    time=sample_time,
                    frame=row.frame,
                    game_seconds=game_seconds,
                    overtime=current_features[overtime_index] >= 0.5,
                    actions={},
                ),
            )
            player_id = row.player_id
            if player_id in sample.actions:
                raise ValueError(
                    f"duplicate action for replay {replay_id}, time {sample_time}, "
                    f"player {player_id} in row {row_number}"
                )
            sample.actions[player_id] = PlayerAction(
                player_id=player_id,
                team=row.team,
                label=row.label,
                confidence=row.label_confidence,
                mechanic=row.mechanic,
                mechanic_confidence=row.mechanic_confidence,
                events=row.events,
            )

    return {
        replay_id: sorted(samples.values(), key=lambda sample: sample.time)
        for replay_id, samples in grouped.items()
    }


def player_name_map(replay_json: Path | None) -> dict[str, str]:
    if replay_json is None:
        return {}
    with replay_json.open(encoding="utf-8") as source:
        replay = json.load(source)
    names: dict[str, str] = {}
    for player in replay.get("properties", {}).get("PlayerStats", []):
        name = str(player.get("Name") or "Unknown")
        identity = str(player.get("OnlineID") or name)
        player_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        names[player_id] = name
    return names


def roster_names(
    samples: Iterable[TimelineSample], known_names: dict[str, str]
) -> tuple[list[str], dict[str, str], dict[str, int]]:
    teams: dict[str, int] = {}
    for sample in samples:
        for action in sample.actions.values():
            teams[action.player_id] = action.team
    roster = sorted(teams, key=lambda player_id: (teams[player_id], player_id))
    display_names = dict(known_names)
    team_counts = {0: 0, 1: 0}
    for player_id in roster:
        team = teams[player_id]
        team_counts[team] = team_counts.get(team, 0) + 1
        display_names.setdefault(player_id, f"Player {team_counts[team]}")
    return roster, display_names, teams


def sampled_events(
    samples: list[TimelineSample], interval: float, start: float | None, end: float | None
) -> list[TimelineSample | Gap]:
    selected: list[TimelineSample | Gap] = []
    previous: TimelineSample | None = None
    next_emit: float | None = None
    gap_threshold = max(2.0, interval * 2.0)

    for sample in samples:
        if start is not None and sample.time < start:
            previous = sample
            continue
        if end is not None and sample.time > end:
            break
        if previous is not None and sample.time - previous.time > gap_threshold:
            selected.append(Gap(previous.time, sample.time))
            next_emit = sample.time
        if next_emit is None:
            next_emit = sample.time
        if sample.time + 1e-6 >= next_emit:
            selected.append(sample)
            while next_emit <= sample.time + 1e-6:
                next_emit += interval
        previous = sample
    return selected


def format_duration(seconds: float, tenths: bool = False) -> str:
    seconds = max(0.0, seconds)
    if tenths:
        total_tenths = round(seconds * 10.0)
        minutes, remaining_tenths = divmod(total_tenths, 600)
        return f"{minutes:02d}:{remaining_tenths / 10.0:04.1f}"
    minutes, remaining = divmod(round(seconds), 60)
    return f"{minutes:02d}:{remaining:02d}"


def format_sample(
    sample: TimelineSample,
    roster: list[str],
    names: dict[str, str],
    teams: dict[str, int],
    show_confidence: bool,
) -> str:
    if sample.overtime:
        clock = "OT"
    elif sample.game_seconds is None:
        clock = "--:--"
    else:
        clock = format_duration(sample.game_seconds)
    parts = [f"[{format_duration(sample.time, tenths=True)} elapsed | {clock} game]"]
    for player_id in roster:
        team = teams[player_id]
        action = sample.actions.get(player_id)
        label = ACTION_WORDS.get(action.label, action.label.lower()) if action else "absent"
        confidence = f" {action.confidence:.0%}" if action and show_confidence else ""
        details = label + confidence
        if action:
            mechanic = MECHANIC_WORDS.get(action.mechanic, action.mechanic.lower())
            mechanic_confidence = (
                f" {action.mechanic_confidence:.0%}" if show_confidence else ""
            )
            details += f" / {mechanic}{mechanic_confidence}"
            if action.events:
                event_words = [EVENT_WORDS.get(event, event.lower()) for event in action.events]
                details += f" / {', '.join(event_words)}"
        parts.append(f"{TEAM_NAMES.get(team, f'Team {team}')} {names[player_id]}: {details}")
    return " | ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="preprocessed .sqlite training shard")
    parser.add_argument("--metadata", type=Path, help="metadata manifest; inferred by default")
    parser.add_argument("--replay-json", type=Path, help="raw rrrocket JSON used to restore player names")
    parser.add_argument("--replay-id", help="replay to display when the dataset contains several")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between reports")
    parser.add_argument("--start", type=float, help="starting replay elapsed time")
    parser.add_argument("--end", type=float, help="ending replay elapsed time")
    parser.add_argument("--changes-only", action="store_true", help="only report when an action changes")
    parser.add_argument("--show-confidence", action="store_true")
    parser.add_argument("--realtime", action="store_true", help="wait between reports to replay in real time")
    parser.add_argument("--speed", type=float, default=1.0, help="real-time playback multiplier")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interval <= 0 or args.speed <= 0:
        raise ValueError("interval and speed must be positive")
    dataset = Path(args.dataset).resolve()
    metadata = load_metadata(dataset, args.metadata)
    timelines = load_timeline(dataset, metadata)
    if not timelines:
        raise ValueError("dataset contains no examples")
    if args.replay_id:
        if args.replay_id not in timelines:
            raise ValueError(f"replay ID not found: {args.replay_id}")
        selected_id = args.replay_id
    elif len(timelines) == 1:
        selected_id = next(iter(timelines))
    else:
        available = ", ".join(sorted(timelines))
        raise ValueError(f"dataset contains multiple replays; use --replay-id ({available})")

    samples = timelines[selected_id]
    roster, names, teams = roster_names(samples, player_name_map(args.replay_json))
    events = sampled_events(samples, args.interval, args.start, args.end)
    previous_labels: tuple[tuple[str, str, str, tuple[str, ...]], ...] | None = None
    previous_time: float | None = None
    print(f"Replay {selected_id}: {len(roster)} players")
    for event in events:
        if isinstance(event, Gap):
            if args.realtime and previous_time is not None:
                time_module.sleep(max(0.0, event.end - previous_time) / args.speed)
            print(
                f"[{format_duration(event.start, True)}-{format_duration(event.end, True)} elapsed] "
                "excluded window: goal replay, countdown, or label boundary"
            )
            previous_time = event.end
            previous_labels = None
            continue
        labels = tuple(
            (
                player_id,
                event.actions[player_id].label,
                event.actions[player_id].mechanic,
                event.actions[player_id].events,
            )
            if player_id in event.actions
            else (player_id, "ABSENT", "ABSENT", ())
            for player_id in roster
        )
        if args.changes_only and labels == previous_labels:
            continue
        if args.realtime and previous_time is not None:
            time_module.sleep(max(0.0, event.time - previous_time) / args.speed)
        print(format_sample(event, roster, names, teams, args.show_confidence), flush=True)
        previous_time = event.time
        previous_labels = labels
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
