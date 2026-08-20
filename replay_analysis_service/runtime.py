"""Replay loading and JSON serialization helpers for the analysis service."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .predictor import RollingIntentPredictor
from .reconstruction import (
    LabelConfig,
    ReplayReconstructor,
    WorldState,
    canonical_vector,
    stable_player_id,
)


ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "prediction-model" / "artifacts"
TEAM_NAMES = {0: "Blue", 1: "Orange"}
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "intent_gru.pt"
TACTICAL_CHECKPOINTS = [
    CHECKPOINT_DIR / "intent_gru_0_1.pt",
    CHECKPOINT_DIR / "intent_gru_1_2.pt",
    CHECKPOINT_DIR / "intent_gru_2_3_5.pt",
]


def parse_replay(executable: Path, replay: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("wb") as output:
            result = subprocess.run(
                [str(executable), "--network-parse", str(replay)],
                stdout=output,
                stderr=subprocess.PIPE,
                check=False,
            )
        if result.returncode != 0:
            message = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(message or f"rrrocket exited with status {result.returncode}")
        with temporary.open(encoding="utf-8") as source:
            parsed = json.load(source)
        if "network_frames" not in parsed:
            raise ValueError("rrrocket output does not contain network frames")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_replay(path: Path, progress_stream: Any | None = None) -> dict[str, Any]:
    if progress_stream is None:
        progress_stream = sys.stdout
    if not path.is_file():
        raise FileNotFoundError(f"replay not found: {path}")
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as source:
            replay = json.load(source)
    elif path.suffix.lower() == ".replay":
        executable = ROOT / "rrrocket"
        if not executable.is_file():
            raise FileNotFoundError(f"rrrocket executable not found: {executable}")
        with tempfile.TemporaryDirectory(prefix="replay-analysis-") as temporary:
            parsed = Path(temporary) / f"{path.stem}.json"
            print(f"Parsing {path.name}...", file=progress_stream, flush=True)
            parse_replay(executable, path, parsed)
            with parsed.open(encoding="utf-8") as source:
                replay = json.load(source)
    else:
        raise ValueError("input must be a .replay file or parsed .json file")
    if "network_frames" not in replay:
        raise ValueError("parsed replay does not contain network frames")
    return replay


def player_names(reconstructor: ReplayReconstructor) -> dict[str, str]:
    names: dict[str, str] = {}
    for actor in reconstructor.actors.values():
        name = actor.properties.get("Engine.PlayerReplicationInfo:PlayerName")
        if name:
            names[stable_player_id(actor)] = str(name)
    return names


def format_duration(seconds: float) -> str:
    minutes, remaining = divmod(max(0, round(seconds)), 60)
    return f"{minutes:02d}:{remaining:02d}"


def game_clock(world: WorldState) -> str:
    if world.overtime:
        return "OT"
    if world.seconds_remaining is None:
        return "--:--"
    return format_duration(world.seconds_remaining)


def crossed_goal(goal_frames: list[int], previous_frame: int, frame: int) -> bool:
    return any(previous_frame < goal_frame <= frame for goal_frame in goal_frames)


def window_name(predictor: RollingIntentPredictor) -> str:
    start = float(predictor.target_window["start_seconds"])
    end = float(predictor.target_window["end_seconds"])
    return f"{start:g}-{end:g}s"


def label_config(predictor: RollingIntentPredictor) -> LabelConfig:
    saved = predictor.label_config
    start = float(predictor.target_window["start_seconds"])
    end = float(predictor.target_window["end_seconds"])
    return LabelConfig(
        horizon_seconds=end - start,
        challenge_distance=float(saved.get("challenge_distance", 350.0)),
        challenge_progress=float(saved.get("challenge_progress", 300.0)),
        rotation_distance=float(saved.get("rotation_distance", 400.0)),
        rotation_backtrack=float(saved.get("rotation_backtrack", 300.0)),
        possession_distance=float(saved.get("possession_distance", 450.0)),
        hold_displacement=float(saved.get("hold_displacement", 200.0)),
        airborne_height=float(saved.get("airborne_height", 100.0)),
        ball_contact_distance=float(saved.get("ball_contact_distance", 220.0)),
        boost_pickup_gain=float(saved.get("boost_pickup_gain", 0.1)),
    )


def vector_json(value: tuple[float, float, float]) -> dict[str, float]:
    return {"x": float(value[0]), "y": float(value[1]), "z": float(value[2])}


def prediction_horizon_json(
    predictor: RollingIntentPredictor,
    prediction: Any,
    team: int,
    anchor_seconds: float,
) -> dict[str, Any]:
    world_position = canonical_vector(prediction.position, team)
    world_forward = canonical_vector(prediction.forward, team)
    return {
        "window": {
            "startSeconds": float(predictor.target_window["start_seconds"]),
            "endSeconds": float(predictor.target_window["end_seconds"]),
        },
        "targetSeconds": float(anchor_seconds + prediction.seconds),
        "position": vector_json(world_position),
        "forward": vector_json(world_forward),
    }


def player_sample_json(
    anchor_seconds: float,
    horizons: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "anchorSeconds": float(anchor_seconds),
        "status": "available" if horizons is not None else "unavailable",
        "horizons": horizons or [],
    }
