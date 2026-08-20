import json
import sqlite3
import tempfile
import unittest
from array import array
from pathlib import Path

from training_dataset import ReplayDataset


class ReplayDatasetTests(unittest.TestCase):
    def test_assembles_history_from_one_float32_track(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.sqlite"
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE feature_tracks (
                    id INTEGER PRIMARY KEY, feature_count INTEGER, state_count INTEGER, features BLOB
                );
                CREATE TABLE examples (
                    id INTEGER PRIMARY KEY, replay_id TEXT, time REAL, frame INTEGER,
                    player_id TEXT, team INTEGER, label INTEGER, label_confidence REAL,
                    mechanic INTEGER, mechanic_confidence REAL, event_mask INTEGER,
                    track_id INTEGER, state_index INTEGER
                );
                """
            )
            metadata = {
                "sequence_length": 3,
                "feature_names": ["first", "second"],
                "labels": ["OTHER"],
                "mechanics": ["GROUNDED"],
                "events": ["SHOT", "CLEAR"],
            }
            connection.execute(
                "INSERT INTO metadata VALUES ('schema', ?)",
                (json.dumps(metadata),),
            )
            values = array("f", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            connection.execute(
                "INSERT INTO feature_tracks VALUES (1, 2, 4, ?)",
                (values.tobytes(),),
            )
            connection.execute(
                "INSERT INTO examples VALUES (1, 'replay', 0.2, 20, 'player', 0, 0, 0.75, 0, 0.8, 2, 1, 2)"
            )
            connection.commit()
            connection.close()

            with ReplayDataset(path) as dataset:
                example = dataset[0]
                self.assertEqual(example.shape, (3, 2))
                self.assertEqual(example.features.tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                self.assertEqual(example.current_features.tolist(), [5.0, 6.0])
                self.assertEqual(example.label, "OTHER")
                self.assertEqual(example.events, ("CLEAR",))


if __name__ == "__main__":
    unittest.main()
