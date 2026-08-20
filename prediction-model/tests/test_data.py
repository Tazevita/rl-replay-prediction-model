import sqlite3
from array import array

import pytest

torch = pytest.importorskip("torch")

from prediction_model.data import ReplayWindowDataset, Shard, TrackBatchSampler, split_replays


def test_split_replays_keeps_whole_replays_apart(tmp_path):
    shards = [Shard(tmp_path / f"{index}.sqlite", f"replay-{index}") for index in range(3)]
    train, validation = split_replays(shards, validation_fraction=0.2, seed=7)

    assert len(train) == 2
    assert len(validation) == 1
    assert {item.replay_id for item in train}.isdisjoint(
        item.replay_id for item in validation
    )


def test_dataset_reads_expected_window(tmp_path):
    path = tmp_path / "replay.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE feature_tracks (
            id INTEGER PRIMARY KEY,
            feature_count INTEGER,
            state_count INTEGER,
            features BLOB
        );
        CREATE TABLE examples (
            id INTEGER PRIMARY KEY,
            label INTEGER,
            label_confidence REAL,
            track_id INTEGER,
            state_index INTEGER,
            target_position_x REAL,
            target_position_y REAL,
            target_position_z REAL,
            target_forward_x REAL,
            target_forward_y REAL,
            target_forward_z REAL
        );
        """
    )
    values = array("f", [1, 2, 3, 4, 5, 6, 7, 8])
    connection.execute("INSERT INTO feature_tracks VALUES (1, 2, 4, ?)", (values.tobytes(),))
    connection.execute(
        "INSERT INTO examples VALUES (1, 2, 0.75, 1, 2, 0.25, -0.5, 0.1, 0, 1, 0)"
    )
    connection.commit()
    connection.close()

    dataset = ReplayWindowDataset([Shard(path, "replay")], 3, 2, cache_tracks=1)
    try:
        features, label, confidence, position, forward = dataset[0]
        assert features.tolist() == [[1, 2], [3, 4], [5, 6]]
        assert label == 2
        assert confidence == pytest.approx(0.75)
        assert position.tolist() == pytest.approx([0.25, -0.5, 0.1])
        assert forward.tolist() == pytest.approx([0, 1, 0])
    finally:
        dataset.close()


def test_track_batch_sampler_keeps_track_reads_adjacent(tmp_path):
    path = tmp_path / "replay.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE feature_tracks (
            id INTEGER PRIMARY KEY,
            feature_count INTEGER,
            state_count INTEGER,
            features BLOB
        );
        CREATE TABLE examples (
            id INTEGER PRIMARY KEY,
            label INTEGER,
            label_confidence REAL,
            track_id INTEGER,
            state_index INTEGER,
            target_position_x REAL,
            target_position_y REAL,
            target_position_z REAL,
            target_forward_x REAL,
            target_forward_y REAL,
            target_forward_z REAL
        );
        """
    )
    values = array("f", range(8))
    for track_id in (1, 2):
        connection.execute(
            "INSERT INTO feature_tracks VALUES (?, 2, 4, ?)",
            (track_id, values.tobytes()),
        )
    for example_id, track_id in enumerate((1, 2, 1, 2, 1, 2), start=1):
        connection.execute(
            "INSERT INTO examples VALUES (?, 0, 1.0, ?, 2, 0, 0, 0, 1, 0, 0)",
            (example_id, track_id),
        )
    connection.commit()
    connection.close()

    dataset = ReplayWindowDataset([Shard(path, "replay")], 1, 2, cache_tracks=1)
    sampler = TrackBatchSampler(dataset, batch_size=2, shuffle=True, seed=7)
    try:
        batches = list(sampler)
        assert sorted(index for batch in batches for index in batch) == list(range(6))
        assert all(len({dataset.refs[index].track_id for index in batch}) == 1 for batch in batches)
        track_order = [dataset.refs[batch[0]].track_id for batch in batches]
        assert track_order in ([1, 1, 2, 2], [2, 2, 1, 1])
    finally:
        dataset.close()
