"""Read compact replay training shards and assemble feature windows."""

from __future__ import annotations

import json
import sqlite3
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class TrainingExample:
    replay_id: str
    time: float
    frame: int
    player_id: str
    team: int
    label: str
    label_confidence: float
    mechanic: str
    mechanic_confidence: float
    events: tuple[str, ...]
    target_position: tuple[float, float, float] | None
    target_forward: tuple[float, float, float] | None
    features: memoryview
    sequence_length: int
    feature_count: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.sequence_length, self.feature_count)

    @property
    def current_features(self) -> memoryview:
        return self.features[-self.feature_count :]


class ReplayDataset:
    """Random-access loader for SQLite shards produced by preprocess_replays.py."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        metadata_row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema'"
        ).fetchone()
        if metadata_row is None:
            self.connection.close()
            raise ValueError(f"dataset has no schema metadata: {self.path}")
        self.metadata: dict[str, Any] = json.loads(metadata_row[0])
        self.sequence_length = int(self.metadata["sequence_length"])
        self.feature_count = len(self.metadata["feature_names"])
        self._rows = self.connection.execute(
            """SELECT replay_id, time, frame, player_id, team, label, label_confidence,
                      mechanic, mechanic_confidence, event_mask, track_id, state_index
                 FROM examples ORDER BY id"""
        ).fetchall()
        self._target_rows = None
        if int(self.metadata.get("format_version", 0)) >= 6:
            self._target_rows = self.connection.execute(
                """SELECT target_position_x, target_position_y, target_position_z,
                          target_forward_x, target_forward_y, target_forward_z
                     FROM examples ORDER BY id"""
            ).fetchall()
        self._tracks: dict[int, memoryview] = {}

    def __enter__(self) -> ReplayDataset:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[TrainingExample]:
        for index in range(len(self)):
            yield self[index]

    def _track(self, track_id: int) -> memoryview:
        cached = self._tracks.get(track_id)
        if cached is not None:
            return cached
        row = self.connection.execute(
            "SELECT feature_count, state_count, features FROM feature_tracks WHERE id = ?",
            (track_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"missing feature track {track_id}")
        feature_count, state_count, blob = row
        if (
            feature_count != self.feature_count
            or len(blob) != feature_count * state_count * 4
        ):
            raise ValueError(f"invalid feature track {track_id}")
        if sys.byteorder == "little":
            values = memoryview(blob).cast("f")
        else:
            native = array("f")
            native.frombytes(blob)
            native.byteswap()
            values = memoryview(native)
        self._tracks[track_id] = values
        return values

    def __getitem__(self, index: int) -> TrainingExample:
        row = self._rows[index]
        target_row = self._target_rows[index] if self._target_rows is not None else None
        event_names = self.metadata["events"]
        event_mask = int(row[9])
        end = (int(row[11]) + 1) * self.feature_count
        start = end - self.sequence_length * self.feature_count
        if start < 0:
            raise ValueError(f"example {index} has an incomplete feature window")
        features = self._track(int(row[10]))[start:end]
        return TrainingExample(
            replay_id=str(row[0]),
            time=float(row[1]),
            frame=int(row[2]),
            player_id=str(row[3]),
            team=int(row[4]),
            label=self.metadata["labels"][int(row[5])],
            label_confidence=float(row[6]),
            mechanic=self.metadata["mechanics"][int(row[7])],
            mechanic_confidence=float(row[8]),
            events=tuple(
                name for bit, name in enumerate(event_names) if event_mask & (1 << bit)
            ),
            target_position=(
                tuple(float(value) for value in target_row[:3]) if target_row is not None else None
            ),
            target_forward=(
                tuple(float(value) for value in target_row[3:]) if target_row is not None else None
            ),
            features=features,
            sequence_length=self.sequence_length,
            feature_count=self.feature_count,
        )
