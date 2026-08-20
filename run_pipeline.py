#!/usr/bin/env python3
"""Parse input replays and build the custom action dataset in one command."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "replay_input"
DEFAULT_PARSED = ROOT / "rrrocket_parsed"
DEFAULT_OUTPUT = ROOT / "training_output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--parsed-dir", type=Path, default=DEFAULT_PARSED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-name", default="manifest.json")
    parser.add_argument("--team-size", type=int, default=2)
    parser.add_argument("--target-offset-seconds", type=float, default=0.0)
    parser.add_argument("--horizon-seconds", type=float, default=1.5)
    parser.add_argument(
        "--force", action="store_true", help="reparse every replay even when JSON is current"
    )
    return parser.parse_args()


def parsed_path(replay: Path, input_dir: Path, parsed_dir: Path) -> Path:
    return parsed_dir / replay.relative_to(input_dir).with_suffix(".json")


def shard_path(replay: Path, input_dir: Path, output_dir: Path) -> Path:
    return output_dir / "replays" / replay.relative_to(input_dir).with_suffix(".sqlite")


def legacy_shard_path(replay: Path, input_dir: Path, output_dir: Path) -> Path:
    return output_dir / "replays" / replay.relative_to(input_dir).with_suffix(".jsonl.gz")


def metadata_path(dataset: Path) -> Path:
    return dataset.with_name(f"{dataset.name.removesuffix('.sqlite')}.metadata.json")


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


def current_shard(
    dataset: Path,
    metadata: Path,
    parsed: Path,
    transformer: Path,
    team_size: int,
    target_offset_seconds: float,
    horizon_seconds: float,
) -> bool:
    if not dataset.is_file() or not metadata.is_file():
        return False
    required_mtime = max(parsed.stat().st_mtime, transformer.stat().st_mtime)
    if dataset.stat().st_mtime < required_mtime or metadata.stat().st_mtime < required_mtime:
        return False
    try:
        with metadata.open(encoding="utf-8") as source:
            schema = json.load(source)
            return (
                schema.get("format_version") == 6
                and schema.get("team_size") == team_size
                and schema.get("target_window")
                == {
                    "start_seconds": target_offset_seconds,
                    "end_seconds": target_offset_seconds + horizon_seconds,
                }
            )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    parsed_dir = args.parsed_dir.resolve()
    output_dir = args.output_dir.resolve()
    executable = ROOT / "rrrocket"
    transformer = ROOT / "preprocess_replays.py"

    if not executable.is_file():
        raise FileNotFoundError(f"rrrocket executable not found: {executable}")
    if not transformer.is_file():
        raise FileNotFoundError(f"transform script not found: {transformer}")
    if args.team_size < 1:
        raise ValueError("team size must be at least 1")
    if args.target_offset_seconds < 0 or args.horizon_seconds <= 0:
        raise ValueError("target offset cannot be negative and horizon must be positive")
    endpoint_steps = (args.target_offset_seconds + args.horizon_seconds) * 10.0
    if not math.isclose(endpoint_steps, round(endpoint_steps), abs_tol=1e-6):
        raise ValueError("target window end must align with the 10 Hz sample rate")
    if Path(args.manifest_name).name != args.manifest_name or not args.manifest_name.endswith(
        ".json"
    ):
        raise ValueError("manifest name must be a filename ending in .json")

    input_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    replays = sorted(input_dir.rglob("*.replay"))

    parsed_count = 0
    skipped_count = 0
    failures: list[tuple[Path, str]] = []
    for replay in replays:
        destination = parsed_path(replay, input_dir, parsed_dir)
        if (
            not args.force
            and destination.is_file()
            and destination.stat().st_mtime >= replay.stat().st_mtime
        ):
            skipped_count += 1
            continue
        print(f"Parsing {replay.relative_to(input_dir)}...", flush=True)
        try:
            parse_replay(executable, replay, destination)
            parsed_count += 1
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
            failures.append((replay, str(error)))
            print(f"Failed to parse {replay}: {error}", file=sys.stderr)

    parsed_files = [
        parsed_path(replay, input_dir, parsed_dir)
        for replay in replays
        if parsed_path(replay, input_dir, parsed_dir).is_file()
    ]
    if not parsed_files:
        print(f"No replays found. Add .replay files to {input_dir}", file=sys.stderr)
        return 1
    if failures:
        print("Dataset build stopped because one or more replays failed.", file=sys.stderr)
        return 1

    print(
        f"Parsed {parsed_count}, reused {skipped_count}; transforming {len(parsed_files)} replay(s)...",
        flush=True,
    )
    transformed_count = 0
    reused_shards = 0
    shard_records: list[dict[str, object]] = []
    label_counts: Counter[str] = Counter()
    mechanic_counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    shared_metadata: dict[str, object] | None = None

    for replay, parsed in zip(replays, parsed_files):
        dataset = shard_path(replay, input_dir, output_dir)
        metadata = metadata_path(dataset)
        dataset.parent.mkdir(parents=True, exist_ok=True)
        if args.force or not current_shard(
            dataset,
            metadata,
            parsed,
            transformer,
            args.team_size,
            args.target_offset_seconds,
            args.horizon_seconds,
        ):
            print(f"Transforming {replay.relative_to(input_dir)}...", flush=True)
            result = subprocess.run(
                [
                    sys.executable,
                    str(transformer),
                    str(parsed),
                    "--output",
                    str(dataset),
                    "--team-size",
                    str(args.team_size),
                    "--target-offset-seconds",
                    str(args.target_offset_seconds),
                    "--horizon-seconds",
                    str(args.horizon_seconds),
                ],
                check=False,
            )
            if result.returncode != 0:
                return result.returncode
            transformed_count += 1
        else:
            reused_shards += 1

        legacy_shard_path(replay, input_dir, output_dir).unlink(missing_ok=True)
        with metadata.open(encoding="utf-8") as source:
            shard_metadata = json.load(source)
        replay_summary = shard_metadata["replays"][0]
        label_counts.update(replay_summary.get("labels", {}))
        mechanic_counts.update(replay_summary.get("mechanics", {}))
        event_counts.update(replay_summary.get("events", {}))
        if shared_metadata is None:
            shared_metadata = {
                key: shard_metadata[key]
                for key in (
                    "format_version",
                    "storage",
                    "float_byte_order",
                    "player_slot_assignment",
                    "description",
                    "labels",
                    "mechanics",
                    "events",
                    "feature_names",
                    "sequence_length",
                    "sample_rate_hz",
                    "history_seconds",
                    "team_size",
                    "normalization",
                    "label_config",
                    "target_window",
                    "prediction_targets",
                )
            }
        shard_records.append(
            {
                "replay_id": replay_summary["replay_id"],
                "source_replay": str(replay.relative_to(input_dir)),
                "parsed_json": str(parsed.relative_to(parsed_dir)),
                "dataset": str(dataset.relative_to(output_dir)),
                "metadata": str(metadata.relative_to(output_dir)),
                "examples": replay_summary["examples"],
                "labels": replay_summary.get("labels", {}),
                "mechanics": replay_summary.get("mechanics", {}),
                "events": replay_summary.get("events", {}),
            }
        )

    manifest = {
        "manifest_format_version": 2,
        **(shared_metadata or {}),
        "total_examples": sum(int(record["examples"]) for record in shard_records),
        "label_counts": dict(sorted(label_counts.items())),
        "mechanic_counts": dict(sorted(mechanic_counts.items())),
        "event_counts": dict(sorted(event_counts.items())),
        "shards": shard_records,
    }
    manifest_path = output_dir / args.manifest_name
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    print(
        f"Pipeline complete: transformed {transformed_count}, reused {reused_shards}; "
        f"manifest {manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
