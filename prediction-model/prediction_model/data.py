"""Memory-bounded access to replay preprocessing shards."""

from __future__ import annotations

import json
import random
import sqlite3
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class Shard:
    path: Path
    replay_id: str


@dataclass(frozen=True)
class SampleRef:
    shard: Shard
    example_id: int
    label: int
    confidence: float
    track_id: int
    state_index: int
    target_position: tuple[float, float, float]
    target_forward: tuple[float, float, float]


def load_manifest(path: str | Path) -> tuple[dict[str, Any], list[Shard]]:
    manifest_path = Path(path).resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    required = ("labels", "feature_names", "sequence_length", "shards")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"manifest is missing: {', '.join(missing)}")
    if manifest.get("float_byte_order", "little") != "little":
        raise ValueError("only little-endian float32 datasets are supported")
    if int(manifest.get("format_version", 0)) < 6 or "prediction_targets" not in manifest:
        raise ValueError("dataset must be rebuilt with endpoint prediction targets (format 6 or newer)")
    if sys.byteorder != "little":
        raise RuntimeError("this dataset reader currently requires a little-endian host")

    shards = [
        Shard(
            path=(manifest_path.parent / item["dataset"]).resolve(),
            replay_id=str(item["replay_id"]),
        )
        for item in manifest["shards"]
    ]
    missing_shards = [str(shard.path) for shard in shards if not shard.path.is_file()]
    if missing_shards:
        raise FileNotFoundError(f"dataset shard not found: {missing_shards[0]}")
    return manifest, shards


def split_replays(
    shards: Sequence[Shard], validation_fraction: float = 0.2, seed: int = 7
) -> tuple[list[Shard], list[Shard]]:
    """Split whole replays, never adjacent windows from the same replay."""
    if len(shards) < 2:
        raise ValueError("at least two replay shards are required for validation")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    shuffled = list(shards)
    random.Random(seed).shuffle(shuffled)
    validation_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * validation_fraction)))
    validation_ids = {shard.replay_id for shard in shuffled[:validation_count]}
    train = [shard for shard in shards if shard.replay_id not in validation_ids]
    validation = [shard for shard in shards if shard.replay_id in validation_ids]
    return train, validation


def _read_sample_refs(shards: Sequence[Shard]) -> list[SampleRef]:
    refs: list[SampleRef] = []
    for shard in shards:
        connection = sqlite3.connect(f"file:{shard.path}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                """SELECT id, label, label_confidence, track_id, state_index,
                          target_position_x, target_position_y, target_position_z,
                          target_forward_x, target_forward_y, target_forward_z
                   FROM examples ORDER BY id"""
            )
            refs.extend(
                SampleRef(
                    shard=shard,
                    example_id=int(row[0]),
                    label=int(row[1]),
                    confidence=float(row[2]),
                    track_id=int(row[3]),
                    state_index=int(row[4]),
                    target_position=(float(row[5]), float(row[6]), float(row[7])),
                    target_forward=(float(row[8]), float(row[9]), float(row[10])),
                )
                for row in rows
            )
        finally:
            connection.close()
    return refs


class ReplayWindowDataset(Dataset):
    """Load windows lazily while retaining only a bounded number of tracks."""

    def __init__(
        self,
        shards: Sequence[Shard],
        sequence_length: int,
        feature_count: int,
        cache_tracks: int = 4,
    ) -> None:
        if cache_tracks < 1:
            raise ValueError("cache_tracks must be at least 1")
        self.sequence_length = sequence_length
        self.feature_count = feature_count
        self.cache_tracks = cache_tracks
        self.refs = _read_sample_refs(shards)
        self._connections: dict[Path, sqlite3.Connection] = {}
        self._tracks: OrderedDict[tuple[Path, int], torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return len(self.refs)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_connections"] = {}
        state["_tracks"] = OrderedDict()
        return state

    def close(self) -> None:
        for connection in self._connections.values():
            connection.close()
        self._connections.clear()
        self._tracks.clear()

    def _connection(self, path: Path) -> sqlite3.Connection:
        connection = self._connections.get(path)
        if connection is None:
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            self._connections[path] = connection
        return connection

    def _track(self, ref: SampleRef) -> torch.Tensor:
        key = (ref.shard.path, ref.track_id)
        cached = self._tracks.get(key)
        if cached is not None:
            self._tracks.move_to_end(key)
            return cached

        row = self._connection(ref.shard.path).execute(
            """SELECT feature_count, state_count, features
               FROM feature_tracks WHERE id = ?""",
            (ref.track_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"missing feature track {ref.track_id} in {ref.shard.path}")
        feature_count, state_count, blob = int(row[0]), int(row[1]), row[2]
        expected_bytes = feature_count * state_count * 4
        if feature_count != self.feature_count or len(blob) != expected_bytes:
            raise ValueError(f"invalid feature track {ref.track_id} in {ref.shard.path}")

        # bytearray gives PyTorch a writable, owned buffer and avoids NumPy.
        track = torch.frombuffer(bytearray(blob), dtype=torch.float32).view(
            state_count, feature_count
        )
        self._tracks[key] = track
        if len(self._tracks) > self.cache_tracks:
            self._tracks.popitem(last=False)
        return track

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, int, float, torch.Tensor, torch.Tensor]:
        ref = self.refs[index]
        end = ref.state_index + 1
        start = end - self.sequence_length
        if start < 0:
            raise ValueError(f"example {ref.example_id} has an incomplete history")
        window = self._track(ref)[start:end]
        if window.shape != (self.sequence_length, self.feature_count):
            raise ValueError(f"example {ref.example_id} has an invalid window")
        return (
            window,
            ref.label,
            ref.confidence,
            torch.tensor(ref.target_position, dtype=torch.float32),
            torch.tensor(ref.target_forward, dtype=torch.float32),
        )


class TrackBatchSampler(Sampler[list[int]]):
    """Keep each track's windows adjacent while randomizing training order."""

    def __init__(
        self,
        dataset: ReplayWindowDataset,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.groups: list[list[int]] = []
        group_indices: dict[tuple[Path, int], int] = {}
        for index, ref in enumerate(dataset.refs):
            key = (ref.shard.path, ref.track_id)
            group_index = group_indices.get(key)
            if group_index is None:
                group_index = len(self.groups)
                group_indices[key] = group_index
                self.groups.append([])
            self.groups[group_index].append(index)

    def __iter__(self) -> Iterator[list[int]]:
        groups = list(range(len(self.groups)))
        generator = random.Random(self.seed + self.epoch)
        if self.shuffle:
            generator.shuffle(groups)
            self.epoch += 1

        for group_index in groups:
            indices = self.groups[group_index].copy()
            if self.shuffle:
                generator.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.groups
        )


def label_counts(dataset: ReplayWindowDataset, label_count: int) -> list[int]:
    counts = [0] * label_count
    for ref in dataset.refs:
        if not 0 <= ref.label < label_count:
            raise ValueError(f"label index {ref.label} is outside the manifest schema")
        counts[ref.label] += 1
    return counts
