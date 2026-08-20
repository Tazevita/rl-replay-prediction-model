#!/usr/bin/env python3
"""Reconstruct rrrocket network JSON and derive tactical features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from array import array
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable


POSITION_SCALE = (4096.0, 5120.0, 2044.0)
VELOCITY_SCALE = 2300.0
RRROCKET_ANGULAR_VELOCITY_SCALE = 100.0
ANGULAR_VELOCITY_SCALE = 5.5
INTENT_LABELS = [
    "CHALLENGE",
    "POSSESS",
    "SUPPORT",
    "SHADOW",
    "CLOSE_ROTATE",
    "HOLD",
    "OTHER",
    "BOOST_DETOUR",
    "BUMP",
    "PRESSURE",
    "REPOSITION",
    "DEFEND",
    "FAR_ROTATE",
    "ATTACK",
    "CHERRY_PICK",
]
LARGE_BOOST_PADS = (
    (-3072.0, -4096.0, 73.0),
    (3072.0, -4096.0, 73.0),
    (-3584.0, 0.0, 73.0),
    (3584.0, 0.0, 73.0),
    (-3072.0, 4096.0, 73.0),
    (3072.0, 4096.0, 73.0),
)
SIDE_WALL_X = 4096.0
BACK_WALL_Y = 5120.0
GOAL_HALF_WIDTH = 893.0
BALL_RADIUS = 92.75


@dataclass
class Actor:
    name: str
    object_name: str
    properties: dict[str, Any] = field(default_factory=dict)
    trajectory: dict[str, Any] = field(default_factory=dict)
    last_update_frame: int = -1
    demolished_until: float = -1.0


@dataclass
class BodyState:
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    rotation: tuple[float, float, float, float]


@dataclass
class PlayerState:
    player_id: str
    team: int
    body: BodyState
    boost: float


@dataclass
class WorldState:
    time: float
    frame: int
    seconds_remaining: float | None
    overtime: bool
    game_active: bool
    score: tuple[int, int]
    ball: BodyState | None
    players: dict[str, PlayerState]


@dataclass
class LabelConfig:
    horizon_seconds: float
    challenge_distance: float
    challenge_progress: float
    rotation_distance: float
    rotation_backtrack: float
    possession_distance: float = 450.0
    hold_displacement: float = 200.0
    airborne_height: float = 100.0
    ball_contact_distance: float = 220.0
    boost_pickup_gain: float = 0.1


@dataclass
class ActionClassification:
    intent: str
    intent_confidence: float
    mechanic: str
    mechanic_confidence: float
    events: list[str]


def decode_attribute(attribute: Any) -> Any:
    if not isinstance(attribute, dict) or not attribute:
        return attribute
    return next(iter(attribute.values()))


def actor_reference(value: Any) -> int | None:
    if not isinstance(value, dict) or not value.get("active"):
        return None
    actor = value.get("actor")
    return actor if isinstance(actor, int) else None


def vector(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, dict):
        return (0.0, 0.0, 0.0)
    return tuple(float(value.get(axis) or 0.0) for axis in ("x", "y", "z"))


def rrrocket_angular_velocity(value: Any) -> tuple[float, float, float]:
    x, y, z = vector(value)
    return tuple(
        component / RRROCKET_ANGULAR_VELOCITY_SCALE for component in (x, y, z)
    )


def quaternion(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, dict):
        return (0.0, 0.0, 0.0, 1.0)
    q = tuple(float(value.get(axis) or 0.0) for axis in ("x", "y", "z", "w"))
    length = math.sqrt(sum(component * component for component in q))
    if length < 1e-6:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(component / length for component in q)  # type: ignore[return-value]


def body_from_actor(actor: Actor) -> BodyState | None:
    rigid = actor.properties.get("TAGame.RBActor_TA:ReplicatedRBState")
    if isinstance(rigid, dict) and isinstance(rigid.get("location"), dict):
        return BodyState(
            position=vector(rigid["location"]),
            velocity=vector(rigid.get("linear_velocity")),
            angular_velocity=rrrocket_angular_velocity(rigid.get("angular_velocity")),
            rotation=quaternion(rigid.get("rotation")),
        )

    location = actor.trajectory.get("location")
    if isinstance(location, dict):
        return BodyState(
            position=vector(location),
            velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
        )
    return None


def stable_player_id(pri: Actor) -> str:
    unique = pri.properties.get("Engine.PlayerReplicationInfo:UniqueId")
    identity = ""
    if isinstance(unique, dict):
        remote = unique.get("remote_id")
        if isinstance(remote, dict) and remote:
            identity = str(next(iter(remote.values())))
    if not identity:
        identity = str(pri.properties.get("Engine.PlayerReplicationInfo:PlayerName") or pri.name)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def canonical_xy_sign(team: int) -> float:
    # Team 0 attacks +Y. Rotate Team 1 by 180 degrees so it does too.
    return 1.0 if team == 0 else -1.0


def canonical_vector(value: tuple[float, float, float], team: int) -> tuple[float, float, float]:
    sign = canonical_xy_sign(team)
    return (sign * value[0], sign * value[1], value[2])


def subtract(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


def distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def magnitude(value: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in value))


def reflect_coordinate(value: float, limit: float) -> float:
    while abs(value) > limit:
        value = math.copysign(2.0 * limit - abs(value), value)
    return value


def predict_defensive_ball_position(
    position: tuple[float, float, float],
    velocity: tuple[float, float, float],
    seconds: float,
) -> tuple[float, float, float]:
    side_limit = SIDE_WALL_X - BALL_RADIUS
    back_limit = BACK_WALL_Y - BALL_RADIUS
    predicted_x = reflect_coordinate(position[0] + velocity[0] * seconds, side_limit)
    predicted_y = position[1] + velocity[1] * seconds
    # The back wall reflects the ball except across the goal opening.
    if abs(predicted_x) > GOAL_HALF_WIDTH:
        predicted_y = reflect_coordinate(predicted_y, back_limit)
    return (predicted_x, max(predicted_y, -BACK_WALL_Y), position[2] + velocity[2] * seconds)


def distance_to_defensive_lane(
    point: tuple[float, float, float], threat: tuple[float, float, float]
) -> float:
    goal = (0.0, -BACK_WALL_Y)
    lane_x = threat[0] - goal[0]
    lane_y = threat[1] - goal[1]
    length_squared = lane_x * lane_x + lane_y * lane_y
    if length_squared < 1e-6:
        return math.hypot(point[0] - goal[0], point[1] - goal[1])
    projection = max(
        0.0,
        min(
            1.0,
            ((point[0] - goal[0]) * lane_x + (point[1] - goal[1]) * lane_y)
            / length_squared,
        ),
    )
    closest_x = goal[0] + projection * lane_x
    closest_y = goal[1] + projection * lane_y
    return math.hypot(point[0] - closest_x, point[1] - closest_y)


def orientation_vectors(
    rotation: tuple[float, float, float, float], team: int
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    x, y, z, w = rotation
    forward = (
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + w * z),
        2.0 * (x * z - w * y),
    )
    up = (
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    )
    return canonical_vector(forward, team), canonical_vector(up, team)


class ReplayReconstructor:
    def __init__(self, replay: dict[str, Any], sample_rate: float):
        self.replay = replay
        self.sample_interval = 1.0 / sample_rate
        self.objects: list[str] = replay.get("objects", [])
        self.names: list[str] = replay.get("names", [])
        self.actors: dict[int, Actor] = {}
        self.seconds_remaining: float | None = None
        self.overtime = False
        self.game_active = False
        self.header_teams = {
            str(player.get("Name")): int(player.get("Team", -1))
            for player in replay.get("properties", {}).get("PlayerStats", [])
            if player.get("Name") is not None
        }
        self.goal_frames = sorted(
            int(mark["frame"])
            for mark in replay.get("tick_marks", [])
            if "Goal" in str(mark.get("description", "")) and isinstance(mark.get("frame"), int)
        )
        self.team_goal_frames = {
            team: sorted(
                int(mark["frame"])
                for mark in replay.get("tick_marks", [])
                if mark.get("description") == f"Team{team}Goal" and isinstance(mark.get("frame"), int)
            )
            for team in (0, 1)
        }

    def reconstruct(self) -> list[WorldState]:
        frames = self.replay.get("network_frames", {}).get("frames", [])
        if not frames:
            return []

        worlds: list[WorldState] = []
        next_sample = float(frames[0].get("time", 0.0))
        for frame_index, frame in enumerate(frames):
            self._apply_frame(frame, frame_index)
            frame_time = float(frame.get("time", next_sample))
            while frame_time + 1e-6 >= next_sample:
                worlds.append(self._snapshot(next_sample, frame_index))
                next_sample += self.sample_interval
        return worlds

    def _apply_frame(self, frame: dict[str, Any], frame_index: int) -> None:
        frame_time = float(frame.get("time", 0.0))
        for deleted in frame.get("deleted_actors", []):
            actor_id = deleted.get("actor_id") if isinstance(deleted, dict) else deleted
            if isinstance(actor_id, int):
                self.actors.pop(actor_id, None)

        for new_actor in frame.get("new_actors", []):
            actor_id = new_actor.get("actor_id")
            if not isinstance(actor_id, int):
                continue
            name_id = new_actor.get("name_id")
            object_id = new_actor.get("object_id")
            name = self.names[name_id] if isinstance(name_id, int) and name_id < len(self.names) else ""
            object_name = (
                self.objects[object_id]
                if isinstance(object_id, int) and object_id < len(self.objects)
                else ""
            )
            self.actors[actor_id] = Actor(
                name=name,
                object_name=object_name,
                trajectory=new_actor.get("initial_trajectory") or {},
                last_update_frame=frame_index,
            )

        for update in frame.get("updated_actors", []):
            actor_id = update.get("actor_id")
            object_id = update.get("object_id")
            actor = self.actors.get(actor_id)
            if actor is None or not isinstance(object_id, int) or object_id >= len(self.objects):
                continue
            property_name = self.objects[object_id]
            value = decode_attribute(update.get("attribute"))
            actor.properties[property_name] = value
            actor.last_update_frame = frame_index
            if property_name.endswith(":SecondsRemaining") and isinstance(value, (int, float)):
                self.seconds_remaining = float(value)
            elif property_name.endswith(":bOverTime"):
                self.overtime = bool(value)
            elif property_name.endswith(":ReplicatedStateName"):
                state_name = self.names[value] if isinstance(value, int) and value < len(self.names) else value
                self.game_active = state_name == "Active"
            elif property_name == "TAGame.Car_TA:ReplicatedDemolish":
                # Standard Soccar respawn is three seconds; hide stale car physics meanwhile.
                actor.demolished_until = frame_time + 3.0

    def _snapshot(self, sample_time: float, frame_index: int) -> WorldState:
        boost_by_car: dict[int, float] = {}
        for actor in self.actors.values():
            if not actor.name.startswith("CarComponent_Boost_TA_"):
                continue
            car_id = actor_reference(actor.properties.get("TAGame.CarComponent_TA:Vehicle"))
            amount = actor.properties.get("TAGame.CarComponent_Boost_TA:ReplicatedBoostAmount")
            if car_id is not None and isinstance(amount, (int, float)):
                boost_by_car[car_id] = max(0.0, min(1.0, float(amount) / 255.0))

        players: dict[str, PlayerState] = {}
        ball_candidates: list[tuple[int, BodyState]] = []
        for actor_id, actor in self.actors.items():
            if actor.name.startswith("Ball_TA_"):
                body = body_from_actor(actor)
                if body is not None:
                    ball_candidates.append((actor.last_update_frame, body))
                continue
            if not actor.name.startswith("Car_TA_"):
                continue
            if sample_time < actor.demolished_until:
                continue

            body = body_from_actor(actor)
            pri_id = actor_reference(actor.properties.get("Engine.Pawn:PlayerReplicationInfo"))
            pri = self.actors.get(pri_id) if pri_id is not None else None
            if body is None or pri is None:
                continue
            team_paint = actor.properties.get("TAGame.Car_TA:TeamPaint")
            team = team_paint.get("team") if isinstance(team_paint, dict) else None
            if team not in (0, 1):
                player_name = str(pri.properties.get("Engine.PlayerReplicationInfo:PlayerName") or "")
                team = self.header_teams.get(player_name)
            if team not in (0, 1):
                continue
            player_id = stable_player_id(pri)
            players[player_id] = PlayerState(
                player_id=player_id,
                team=int(team),
                body=body,
                boost=boost_by_car.get(actor_id, 0.0),
            )

        ball = max(ball_candidates, default=(0, None), key=lambda item: item[0])[1]
        score = tuple(
            sum(goal_frame <= frame_index for goal_frame in self.team_goal_frames[team])
            for team in (0, 1)
        )
        return WorldState(
            time=sample_time,
            frame=frame_index,
            seconds_remaining=self.seconds_remaining,
            overtime=self.overtime,
            game_active=self.game_active,
            score=score,  # type: ignore[arg-type]
            ball=ball,
            players=players,
        )


def entity_feature_names(prefix: str) -> list[str]:
    return [
        f"{prefix}_abs_x",
        f"{prefix}_abs_y",
        f"{prefix}_abs_z",
        f"{prefix}_rel_x",
        f"{prefix}_rel_y",
        f"{prefix}_rel_z",
        f"{prefix}_rel_vx",
        f"{prefix}_rel_vy",
        f"{prefix}_rel_vz",
        f"{prefix}_forward_x",
        f"{prefix}_forward_y",
        f"{prefix}_forward_z",
        f"{prefix}_up_x",
        f"{prefix}_up_y",
        f"{prefix}_up_z",
        f"{prefix}_angular_vx",
        f"{prefix}_angular_vy",
        f"{prefix}_angular_vz",
        f"{prefix}_boost",
        f"{prefix}_valid",
    ]


def feature_names(team_size: int) -> list[str]:
    names = [
        "clock_fraction",
        "clock_valid",
        "score_difference",
        "overtime",
        "ball_abs_x",
        "ball_abs_y",
        "ball_abs_z",
        "ball_rel_x",
        "ball_rel_y",
        "ball_rel_z",
        "ball_rel_vx",
        "ball_rel_vy",
        "ball_rel_vz",
        "ball_valid",
    ]
    names.extend(entity_feature_names("ego"))
    for index in range(team_size - 1):
        names.extend(entity_feature_names(f"teammate_{index}"))
    for index in range(team_size):
        names.extend(entity_feature_names(f"opponent_{index}"))
    return names


def scaled_position(value: tuple[float, float, float]) -> list[float]:
    return [component / scale for component, scale in zip(value, POSITION_SCALE)]


def entity_features(
    player: PlayerState | None, ego: PlayerState, team: int
) -> list[float]:
    if player is None:
        return [0.0] * 20
    absolute = canonical_vector(player.body.position, team)
    relative = canonical_vector(subtract(player.body.position, ego.body.position), team)
    relative_velocity = canonical_vector(subtract(player.body.velocity, ego.body.velocity), team)
    forward, up = orientation_vectors(player.body.rotation, team)
    angular = canonical_vector(player.body.angular_velocity, team)
    return [
        *scaled_position(absolute),
        *scaled_position(relative),
        *(component / VELOCITY_SCALE for component in relative_velocity),
        *forward,
        *up,
        *(component / ANGULAR_VELOCITY_SCALE for component in angular),
        player.boost,
        1.0,
    ]


def state_features(
    world: WorldState,
    ego_id: str,
    teammate_ids: list[str],
    opponent_ids: list[str],
    team_size: int,
) -> list[float]:
    ego = world.players.get(ego_id)
    if ego is None:
        return [0.0] * len(feature_names(team_size))
    team = ego.team
    clock_valid = world.seconds_remaining is not None
    values: list[float] = [
        max(0.0, min(1.0, (world.seconds_remaining or 0.0) / 300.0)),
        float(clock_valid),
        (world.score[team] - world.score[1 - team]) / 10.0,
        float(world.overtime),
    ]
    if world.ball is None:
        values.extend([0.0] * 10)
    else:
        ball_abs = canonical_vector(world.ball.position, team)
        ball_rel = canonical_vector(subtract(world.ball.position, ego.body.position), team)
        ball_rel_velocity = canonical_vector(
            subtract(world.ball.velocity, ego.body.velocity), team
        )
        values.extend(
            [
                *scaled_position(ball_abs),
                *scaled_position(ball_rel),
                *(component / VELOCITY_SCALE for component in ball_rel_velocity),
                1.0,
            ]
        )
    values.extend(entity_features(ego, ego, team))
    for player_id in teammate_ids[: team_size - 1]:
        values.extend(entity_features(world.players.get(player_id), ego, team))
    for _ in range(team_size - 1 - len(teammate_ids)):
        values.extend(entity_features(None, ego, team))
    for player_id in opponent_ids[:team_size]:
        values.extend(entity_features(world.players.get(player_id), ego, team))
    for _ in range(team_size - len(opponent_ids)):
        values.extend(entity_features(None, ego, team))
    return values


def stable_team_rosters(
    worlds: list[WorldState], team_size: int
) -> dict[int, list[str]]:
    player_teams: dict[str, int] = {}
    for world in worlds:
        for player_id, player in world.players.items():
            previous_team = player_teams.setdefault(player_id, player.team)
            if previous_team != player.team:
                raise ValueError(f"player {player_id} changed teams during the replay")

    rosters = {
        team: sorted(
            player_id for player_id, player_team in player_teams.items() if player_team == team
        )
        for team in (0, 1)
    }
    for team, roster in rosters.items():
        if len(roster) > team_size:
            raise ValueError(
                f"team {team} has {len(roster)} unique players; "
                f"only {team_size} stable slots are available"
            )
    return rosters


def classify_action(
    worlds: list[WorldState],
    index: int,
    ego_id: str,
    config: LabelConfig,
    end_index: int | None = None,
) -> ActionClassification | None:
    current = worlds[index]
    ego = current.players.get(ego_id)
    if ego is None or current.ball is None:
        return None

    stop = len(worlds) if end_index is None else min(end_index, len(worlds))
    future: list[WorldState] = []
    for future_index in range(index + 1, stop):
        world = worlds[future_index]
        if world.time - current.time > config.horizon_seconds + 1e-6:
            break
        if not world.game_active:
            return None
        if ego_id in world.players and world.ball is not None:
            future.append(world)
    if not future:
        return None
    sample_interval = worlds[index + 1].time - current.time if index + 1 < stop else 0.0
    if future[-1].time - current.time < config.horizon_seconds - sample_interval - 1e-6:
        return None

    future_distances = [
        distance(world.players[ego_id].body.position, world.ball.position)  # type: ignore[union-attr]
        for world in future
    ]
    start_distance = distance(ego.body.position, current.ball.position)
    min_distance = min(future_distances)
    end_ego = future[-1].players[ego_id]
    displacement = distance(ego.body.position, end_ego.body.position)
    progress = start_distance - min_distance
    team = ego.team
    start_position = canonical_vector(ego.body.position, team)
    end_position = canonical_vector(end_ego.body.position, team)
    ball_position = canonical_vector(current.ball.position, team)
    ball_velocity = canonical_vector(current.ball.velocity, team)
    end_ball_position = canonical_vector(future[-1].ball.position, team)  # type: ignore[union-attr]
    lateral = end_position[0] - start_position[0]
    backtrack = start_position[1] - end_position[1]
    advance = end_position[1] - start_position[1]
    ball_advance = end_ball_position[1] - ball_position[1]
    trajectory = [
        start_position,
        *[
            canonical_vector(world.players[ego_id].body.position, team)
            for world in future
        ],
    ]
    path_lateral = sum(
        abs(end[0] - start[0]) for start, end in zip(trajectory, trajectory[1:])
    )
    path_longitudinal = sum(
        abs(end[1] - start[1]) for start, end in zip(trajectory, trajectory[1:])
    )
    path_distance = sum(
        math.hypot(end[0] - start[0], end[1] - start[1])
        for start, end in zip(trajectory, trajectory[1:])
    )
    path_backtrack = sum(
        max(0.0, start[1] - end[1])
        for start, end in zip(trajectory, trajectory[1:])
    )

    possession_samples = [(ego.body, current.ball), *[
        (world.players[ego_id].body, world.ball)  # type: ignore[arg-type]
        for world in future
    ]]
    coupled_samples: list[bool] = []
    for car_body, ball_body in possession_samples:
        sample_distance = distance(car_body.position, ball_body.position)
        airborne = car_body.position[2] >= 150.0 or ball_body.position[2] >= 250.0
        relative_speed_limit = 1600.0 if airborne else 1000.0
        coupled_samples.append(
            sample_distance <= config.possession_distance
            and magnitude(subtract(car_body.velocity, ball_body.velocity))
            <= relative_speed_limit
        )

    # Preserve a carry through one touch impulse, but not through scattered proximity.
    longest_coupled_run = 0
    current_coupled_run = 0
    interruption_used = False
    for coupled in coupled_samples:
        if coupled:
            current_coupled_run += 1
            longest_coupled_run = max(longest_coupled_run, current_coupled_run)
        elif current_coupled_run > 0 and not interruption_used:
            interruption_used = True
        else:
            current_coupled_run = 0
            interruption_used = False

    end_ball = future[-1].ball
    ball_displacement = distance(current.ball.position, end_ball.position)  # type: ignore[union-attr]
    coupled_samples_count = sum(coupled_samples)
    opponents = [player for player in current.players.values() if player.team != team]
    ball_carrier_ids = {
        opponent.player_id
        for opponent in opponents
        if distance(opponent.body.position, current.ball.position)
        <= config.possession_distance * 1.5
    }
    opponent_has_possession = any(
        distance(opponent.body.position, current.ball.position)
        <= config.possession_distance * 1.25
        and magnitude(subtract(opponent.body.velocity, current.ball.velocity)) <= 900.0
        for opponent in opponents
    )
    prediction_seconds = min(config.horizon_seconds, 1.25)
    predicted_ball_position = predict_defensive_ball_position(
        ball_position, ball_velocity, prediction_seconds
    )
    defensive_lane_distance = distance_to_defensive_lane(
        start_position, predicted_ball_position
    )
    end_defensive_lane_distance = distance_to_defensive_lane(
        end_position, predicted_ball_position
    )
    defensive_lane_progress = defensive_lane_distance - end_defensive_lane_distance
    goal = (0.0, -BACK_WALL_Y, 0.0)
    goal_distances = [distance(position, goal) for position in trajectory]
    goal_progress = goal_distances[0] - min(goal_distances[1:])

    ball_samples = [current.ball, *[world.ball for world in future]]
    distance_drops = [
        start - end
        for start, end in zip([start_distance, *future_distances], future_distances)
    ]
    sustained_closing = all(drop >= 50.0 for drop in distance_drops)
    end_forward, _ = orientation_vectors(end_ego.body.rotation, team)
    end_ball_direction = canonical_vector(
        subtract(ball_samples[-1].position, end_ego.body.position), team  # type: ignore[union-attr]
    )
    end_ball_horizontal_distance = math.hypot(
        end_ball_direction[0], end_ball_direction[1]
    )
    end_ball_facing = (
        (end_forward[0] * end_ball_direction[0] + end_forward[1] * end_ball_direction[1])
        / end_ball_horizontal_distance
        if end_ball_horizontal_distance > 0.0
        else 0.0
    )
    timed_states = [current, *future]
    approach_speeds: list[float] = []
    for sample_index, (path_start, path_end) in enumerate(
        zip(trajectory, trajectory[1:]), 1
    ):
        step_seconds = timed_states[sample_index].time - timed_states[sample_index - 1].time
        step = subtract(path_end, path_start)
        sample_ball_direction = canonical_vector(
            subtract(
                ball_samples[sample_index].position,  # type: ignore[union-attr]
                timed_states[sample_index].players[ego_id].body.position,
            ),
            team,
        )
        sample_ball_distance = math.hypot(
            sample_ball_direction[0], sample_ball_direction[1]
        )
        approach_speeds.append(
            (
                (
                    step[0] * sample_ball_direction[0]
                    + step[1] * sample_ball_direction[1]
                )
                / sample_ball_distance
                / step_seconds
                if sample_ball_distance > 0.0 and step_seconds > 0.0
                else 0.0
            )
        )
    accelerating_approach = (
        bool(approach_speeds)
        and approach_speeds[-1] >= 400.0
        and approach_speeds[-1] >= approach_speeds[0] + 100.0
    )
    committed_challenge = (
        displacement >= 300.0
        and progress >= config.challenge_progress
        and min_distance > config.challenge_distance
        and sustained_closing
        and end_ball_facing >= 0.6
        and accelerating_approach
    )
    wide_rotation = (
        path_distance >= config.rotation_distance
        and path_backtrack >= 100.0
        and goal_progress >= 250.0
        and start_position[1] <= ball_position[1] - 500.0
        and min_distance > config.challenge_distance
    )
    rotating = (
        displacement >= config.rotation_distance
        and min_distance > config.challenge_distance
        and (backtrack >= config.rotation_backtrack or wide_rotation)
    )
    entering_goal = (
        end_position[1] <= -BACK_WALL_Y
        and abs(end_position[0]) <= GOAL_HALF_WIDTH + 300.0
        and end_position[1] < start_position[1]
    )
    covering_defensive_lane = (
        ball_position[1] <= 1500.0
        and end_position[1] <= -3000.0
        and defensive_lane_progress >= 400.0
        and end_defensive_lane_distance <= 1000.0
    )

    opponent_chases: list[
        tuple[PlayerState, float, float, float, float, float, float, float, bool]
    ] = []
    for opponent in opponents:
        start_opponent_distance = distance(ego.body.position, opponent.body.position)
        opponent_samples = [
            (world, world.players[opponent.player_id])
            for world in future
            if opponent.player_id in world.players
        ]
        if opponent_samples:
            closest_world, closest_opponent = min(
                opponent_samples,
                key=lambda sample: distance(
                    sample[0].players[ego_id].body.position, sample[1].body.position
                ),
            )
            closest_ego = closest_world.players[ego_id]
            closest_opponent_distance = distance(
                closest_ego.body.position, closest_opponent.body.position
            )
            forward, _ = orientation_vectors(closest_ego.body.rotation, team)
            toward_opponent = canonical_vector(
                subtract(closest_opponent.body.position, closest_ego.body.position), team
            )
            horizontal_distance = math.hypot(toward_opponent[0], toward_opponent[1])
            facing_opponent = (
                (forward[0] * toward_opponent[0] + forward[1] * toward_opponent[1])
                / horizontal_distance
                if horizontal_distance > 0.0
                else 0.0
            )
            toward_opponent_unit = (
                toward_opponent[0] / horizontal_distance,
                toward_opponent[1] / horizontal_distance,
            ) if horizontal_distance > 0.0 else (0.0, 0.0)
            ego_velocity = canonical_vector(closest_ego.body.velocity, team)
            opponent_velocity = canonical_vector(closest_opponent.body.velocity, team)
            ego_opponent_approach = (
                ego_velocity[0] * toward_opponent_unit[0]
                + ego_velocity[1] * toward_opponent_unit[1]
            )
            opponent_ego_approach = -(
                opponent_velocity[0] * toward_opponent_unit[0]
                + opponent_velocity[1] * toward_opponent_unit[1]
            )
            toward_ball = canonical_vector(
                subtract(closest_world.ball.position, closest_ego.body.position), team  # type: ignore[union-attr]
            )
            ball_horizontal_distance = math.hypot(toward_ball[0], toward_ball[1])
            toward_ball_unit = (
                toward_ball[0] / ball_horizontal_distance,
                toward_ball[1] / ball_horizontal_distance,
            ) if ball_horizontal_distance > 0.0 else (0.0, 0.0)
            facing_ball = (
                forward[0] * toward_ball_unit[0] + forward[1] * toward_ball_unit[1]
            )
            ego_ball_approach = (
                ego_velocity[0] * toward_ball_unit[0]
                + ego_velocity[1] * toward_ball_unit[1]
            )
            near_ball = ball_horizontal_distance <= config.possession_distance * 1.5
            opponent_chases.append(
                (
                    opponent,
                    closest_opponent_distance,
                    start_opponent_distance - closest_opponent_distance,
                    facing_opponent,
                    ego_opponent_approach,
                    opponent_ego_approach,
                    facing_ball,
                    ego_ball_approach,
                    near_ball,
                )
            )

    bump_chase = next(
        (
            chase
            for chase in sorted(opponent_chases, key=lambda item: item[1])
            if chase[1] <= 250.0
            and chase[3] >= 0.6
            and chase[4] >= 300.0
            and chase[4] >= chase[5] + 100.0
            and (
                not chase[8]
                or (
                    chase[3] >= chase[6] + 0.15
                    and chase[4] >= chase[7] + 100.0
                )
            )
        ),
        None,
    )

    boost_routes: list[tuple[float, float, tuple[float, float, float]]] = []
    canonical_future_positions = [
        canonical_vector(world.players[ego_id].body.position, team) for world in future
    ]
    for pad in LARGE_BOOST_PADS:
        start_pad_distance = distance(start_position, pad)
        closest_pad_distance = min(distance(position, pad) for position in canonical_future_positions)
        boost_routes.append((closest_pad_distance, start_pad_distance - closest_pad_distance, pad))
    closest_pad_distance, boost_route_progress, target_pad = min(boost_routes)
    pad_is_away_from_play = distance(target_pad, ball_position) >= 1800.0
    if (
        start_distance <= config.possession_distance
        and max(displacement, ball_displacement) >= 150.0
        and longest_coupled_run
        >= max(2, math.ceil(len(possession_samples) * 0.5))
    ):
        intent = "POSSESS"
        intent_confidence = min(
            1.0, 0.6 + coupled_samples_count / len(possession_samples) * 0.4
        )
    elif (
        displacement >= 150.0
        and min_distance <= config.challenge_distance
        and progress >= config.challenge_progress
        and start_distance > 250.0
        and bump_chase is None
    ):
        intent = "CHALLENGE"
        intent_confidence = min(1.0, 0.5 + progress / 2000.0)
    elif committed_challenge and bump_chase is None:
        intent = "CHALLENGE"
        intent_confidence = min(
            1.0, 0.55 + progress / 2000.0 + end_ball_facing * 0.1
        )
    elif (
        opponent_has_possession
        and ball_velocity[1] <= -150.0
        and -BACK_WALL_Y <= start_position[1] <= ball_position[1]
        and abs(start_position[0]) <= SIDE_WALL_X
        and ball_position[1] <= 1500.0
        and 500.0 <= start_distance <= 2800.0
        and defensive_lane_distance <= 1600.0
        and backtrack >= 100.0
        and min_distance > config.challenge_distance
    ):
        intent = "SHADOW"
        incoming_speed = min(1.0, -ball_velocity[1] / VELOCITY_SCALE)
        lane_quality = max(0.0, 1.0 - defensive_lane_distance / 1600.0)
        intent_confidence = min(
            1.0, 0.5 + backtrack / 3000.0 + incoming_speed * 0.15 + lane_quality * 0.15
        )
    else:
        pressure_chase = next(
            (
                chase
                for chase in sorted(opponent_chases, key=lambda item: item[1])
                if chase[0].player_id in ball_carrier_ids
                and chase[1] <= 1200.0
                and chase[2] >= 250.0
            ),
            None,
        )
        teammates = [
            player
            for player in current.players.values()
            if player.team == team and player.player_id != ego_id
        ]
        support_score = 0.0
        cherry_pick_score = 0.0
        for teammate in teammates:
            teammate_samples = [
                (world, world.players[teammate.player_id], world.ball)
                for world in timed_states
                if teammate.player_id in world.players and world.ball is not None
            ]
            if len(teammate_samples) < max(2, math.ceil(len(timed_states) * 0.5)):
                continue

            engaged_samples = 0
            covered_samples = 0
            spacing_samples = 0
            cherry_pick_samples = 0
            for sample_world, sample_teammate, sample_ball in teammate_samples:
                sample_ego = sample_world.players[ego_id]
                ego_ball_distance = distance(
                    sample_ego.body.position, sample_ball.position  # type: ignore[union-attr]
                )
                teammate_ball_distance = distance(
                    sample_teammate.body.position, sample_ball.position  # type: ignore[union-attr]
                )
                teammate_airborne = (
                    sample_teammate.body.position[2] >= 150.0
                    or sample_ball.position[2] >= 250.0  # type: ignore[union-attr]
                )
                engagement_distance = 1600.0 if teammate_airborne else 1200.0
                if (
                    teammate_ball_distance <= engagement_distance
                    and teammate_ball_distance <= ego_ball_distance + 300.0
                ):
                    engaged_samples += 1

                ego_goal_distance = distance(
                    canonical_vector(sample_ego.body.position, team), goal
                )
                teammate_goal_distance = distance(
                    canonical_vector(sample_teammate.body.position, team), goal
                )
                if ego_goal_distance + 150.0 < teammate_goal_distance:
                    covered_samples += 1

                spacing_limit = 5200.0 if teammate_airborne else 3800.0
                teammate_spacing = distance(
                    sample_ego.body.position, sample_teammate.body.position
                )
                if 500.0 <= teammate_spacing <= spacing_limit:
                    spacing_samples += 1

                sample_ego_position = canonical_vector(
                    sample_ego.body.position, team
                )
                sample_teammate_position = canonical_vector(
                    sample_teammate.body.position, team
                )
                sample_ball_position = canonical_vector(
                    sample_ball.position, team  # type: ignore[union-attr]
                )
                if (
                    teammate_ball_distance <= engagement_distance
                    and teammate_ball_distance + 300.0 <= ego_ball_distance
                    and sample_ego_position[1] >= sample_ball_position[1] + 500.0
                    and sample_ego_position[1] >= sample_teammate_position[1] + 500.0
                    and sample_ego_position[1] >= 500.0
                    and 700.0 <= teammate_spacing <= 3500.0
                ):
                    cherry_pick_samples += 1

            required_support_samples = max(
                2, math.ceil(len(teammate_samples) * 0.5)
            )
            if (
                engaged_samples >= required_support_samples
                and covered_samples >= required_support_samples
                and spacing_samples >= required_support_samples
            ):
                teammate_support_score = (
                    engaged_samples + covered_samples + spacing_samples
                ) / (len(teammate_samples) * 3.0)
                support_score = max(support_score, teammate_support_score)
            required_cherry_pick_samples = max(
                2, math.ceil(len(teammate_samples) * 0.5)
            )
            if cherry_pick_samples >= required_cherry_pick_samples:
                cherry_pick_score = max(
                    cherry_pick_score,
                    cherry_pick_samples / len(teammate_samples),
                )
        supporting = support_score > 0.0
        cherry_picking = (
            not opponent_has_possession
            and cherry_pick_score > 0.0
            and min_distance > config.challenge_distance
            and path_backtrack <= max(400.0, path_distance * 0.35)
        )
        if rotating:
            intent = (
                "FAR_ROTATE"
                if ball_position[0] * end_position[0] < 0.0
                else "CLOSE_ROTATE"
            )
            intent_confidence = min(1.0, 0.5 + backtrack / 2500.0)
        elif (
            bump_chase is not None
            and path_distance >= 150.0
        ):
            intent = "BUMP"
            intent_confidence = min(
                1.0,
                0.55
                + bump_chase[3] * 0.15
                + min(1.0, bump_chase[4] / VELOCITY_SCALE) * 0.2,
            )
        elif (
            pressure_chase is not None
            and displacement >= 150.0
            and progress >= 150.0
            and min_distance > config.challenge_distance
        ):
            intent = "PRESSURE"
            intent_confidence = min(1.0, 0.55 + progress / 2000.0)
        elif (
            ego.boost <= 0.75
            and closest_pad_distance <= 450.0
            and boost_route_progress >= 600.0
            and pad_is_away_from_play
            and min_distance > config.challenge_distance
        ):
            intent = "BOOST_DETOUR"
            intent_confidence = min(1.0, 0.55 + boost_route_progress / 3000.0)
        elif cherry_picking:
            intent = "CHERRY_PICK"
            intent_confidence = min(1.0, 0.55 + cherry_pick_score * 0.35)
        elif (
            supporting
            and path_backtrack <= max(500.0, path_distance * 0.5)
            and min_distance > config.challenge_distance
        ):
            intent = "SUPPORT"
            intent_confidence = min(1.0, 0.55 + support_score * 0.35)
        elif (
            not opponent_has_possession
            and displacement >= 400.0
            and advance >= 400.0
            and ball_advance >= 300.0
            and ball_position[1] >= start_position[1] + 250.0
            and end_ball_position[1] >= end_position[1] + 250.0
            and future_distances[-1] <= start_distance + 250.0
            and min_distance > config.challenge_distance
            and start_distance <= 3000.0
            and end_ball_facing >= 0.5
        ):
            intent = "ATTACK"
            intent_confidence = min(
                1.0,
                0.55
                + min(1.0, advance / 2500.0) * 0.2
                + max(0.0, end_ball_facing) * 0.15,
            )
        elif entering_goal or covering_defensive_lane or (
            start_position[1] <= -4200.0
            and end_position[1] <= -4000.0
            and abs(start_position[0]) <= 1500.0
            and abs(end_position[0]) <= 1500.0
            and displacement <= config.hold_displacement * 1.5
        ):
            intent = "DEFEND"
            intent_confidence = min(
                1.0,
                0.65
                + max(0.0, -4200.0 - start_position[1]) / 3000.0
                + max(0.0, defensive_lane_progress) / 4000.0,
            )
        elif (
            path_distance >= 300.0
            and path_lateral >= 300.0
            and path_lateral >= path_longitudinal * 0.75
            and min_distance > config.challenge_distance
        ):
            intent = "REPOSITION"
            intent_confidence = min(1.0, 0.55 + path_lateral / 3000.0)
        elif displacement <= config.hold_displacement:
            intent = "HOLD"
            intent_confidence = min(
                1.0, 0.6 + (config.hold_displacement - displacement) / 500.0
            )
        else:
            intent = "OTHER"
            intent_confidence = 0.5

    _, up = orientation_vectors(ego.body.rotation, team)
    end_height = end_ego.body.position[2]
    if up[2] < 0.65 or (
        ego.body.position[2] > config.airborne_height
        and ego.body.velocity[2] < -150.0
        and end_height < ego.body.position[2] - 100.0
    ):
        mechanic = "RECOVERING"
        mechanic_confidence = min(1.0, 0.65 + max(0.0, 0.65 - up[2]) * 0.35)
    elif ego.body.position[2] > config.airborne_height:
        mechanic = "AERIAL"
        mechanic_confidence = min(
            1.0, 0.6 + (ego.body.position[2] - config.airborne_height) / 1000.0
        )
    else:
        mechanic = "GROUNDED"
        mechanic_confidence = min(
            1.0, 0.7 + (config.airborne_height - ego.body.position[2]) / 500.0
        )

    events: list[str] = []
    states = [current, *future]
    contact_index = min(
        range(len(states)),
        key=lambda state_index: distance(
            states[state_index].players[ego_id].body.position,
            states[state_index].ball.position,  # type: ignore[union-attr]
        ),
    )
    contact_state = states[contact_index]
    contact_distance = distance(
        contact_state.players[ego_id].body.position,
        contact_state.ball.position,  # type: ignore[union-attr]
    )
    if contact_distance <= config.ball_contact_distance and contact_index < len(states) - 1:
        contact_ball = canonical_vector(contact_state.ball.position, team)  # type: ignore[union-attr]
        end_ball = canonical_vector(states[-1].ball.position, team)  # type: ignore[union-attr]
        end_ball_velocity = canonical_vector(states[-1].ball.velocity, team)  # type: ignore[union-attr]
        ball_advance = end_ball[1] - contact_ball[1]
        if ball_advance >= 400.0 or end_ball_velocity[1] >= 700.0:
            events.append("CLEAR" if contact_ball[1] < -1000.0 else "SHOT")

    if max(world.players[ego_id].boost for world in future) - ego.boost >= config.boost_pickup_gain:
        events.append("BOOST_PICKUP")

    for opponent in opponents:
        opponent_states = [world.players.get(opponent.player_id) for world in future]
        if any(player is None for player in opponent_states):
            closest = min(
                [distance(ego.body.position, opponent.body.position)]
                + [
                    distance(world.players[ego_id].body.position, player.body.position)
                    for world, player in zip(future, opponent_states)
                    if player is not None
                ]
            )
            if closest <= 250.0:
                events.append("DEMOLITION")
                break

    return ActionClassification(
        intent, intent_confidence, mechanic, mechanic_confidence, events
    )


def label_action(
    worlds: list[WorldState], index: int, ego_id: str, config: LabelConfig
) -> tuple[str, float] | None:
    classification = classify_action(worlds, index, ego_id, config)
    if classification is None:
        return None
    return (classification.intent, classification.intent_confidence)


def classify_target_window(
    worlds: list[WorldState],
    anchor_index: int,
    ego_id: str,
    config: LabelConfig,
    offset_seconds: float = 0.0,
    goal_frames: list[int] | None = None,
    clip_at_goal: bool = False,
) -> ActionClassification | None:
    """Classify a future interval while keeping model features anchored earlier."""
    target_time = worlds[anchor_index].time + offset_seconds
    for target_index in range(anchor_index, len(worlds)):
        if worlds[target_index].time + 1e-6 >= target_time:
            if abs(worlds[target_index].time - target_time) > 1e-4:
                return None
            if not all(
                worlds[index].game_active
                for index in range(anchor_index, target_index + 1)
            ):
                return None
            if goal_frames is not None:
                end_frame = worlds[target_index].frame
                end_index = target_index
                for future_index in range(target_index + 1, len(worlds)):
                    future = worlds[future_index]
                    if future.time - target_time > config.horizon_seconds + 1e-6:
                        break
                    end_frame = future.frame
                    end_index = future_index
                if crosses_goal(goal_frames, worlds[anchor_index].frame, end_frame):
                    if not clip_at_goal:
                        return None
                    goal_frame = min(
                        frame
                        for frame in goal_frames
                        if worlds[anchor_index].frame <= frame <= end_frame
                    )
                    if goal_frame <= worlds[target_index].frame:
                        return None
                    end_index = next(
                        (
                            index - 1
                            for index in range(target_index + 1, end_index + 1)
                            if worlds[index].frame >= goal_frame
                        ),
                        end_index,
                    )
                    if end_index <= target_index:
                        return None
                    clipped_config = replace(
                        config,
                        horizon_seconds=worlds[end_index].time - target_time,
                    )
                    return classify_action(
                        worlds,
                        target_index,
                        ego_id,
                        clipped_config,
                        end_index=end_index + 1,
                    )
            return classify_action(worlds, target_index, ego_id, config)
    return None


def endpoint_target(
    worlds: list[WorldState],
    anchor_index: int,
    ego_id: str,
    endpoint_seconds: float,
) -> tuple[list[float], tuple[float, float, float]] | None:
    """Return normalized canonical position and facing at an exact future time."""
    target_time = worlds[anchor_index].time + endpoint_seconds
    for index in range(anchor_index, len(worlds)):
        world = worlds[index]
        if world.time + 1e-6 < target_time:
            continue
        if abs(world.time - target_time) > 1e-4 or not world.game_active:
            return None
        player = world.players.get(ego_id)
        if player is None:
            return None
        position = scaled_position(canonical_vector(player.body.position, player.team))
        forward, _ = orientation_vectors(player.body.rotation, player.team)
        return position, forward
    return None


def crosses_goal(goal_frames: list[int], start_frame: int, end_frame: int) -> bool:
    return any(start_frame <= goal_frame <= end_frame for goal_frame in goal_frames)


def replay_id(replay: dict[str, Any], source: Path) -> str:
    value = replay.get("properties", {}).get("Id")
    return str(value or source.stem)


def examples_for_replay(
    replay: dict[str, Any],
    source: Path,
    sample_rate: float,
    history_seconds: float,
    team_size: int,
    label_config: LabelConfig,
    target_offset_seconds: float = 0.0,
) -> tuple[list[dict[str, Any]], list[array], dict[str, Any]]:
    properties = replay.get("properties", {})
    actual_team_size = properties.get("TeamSize")
    if actual_team_size != team_size:
        raise ValueError(f"expected {team_size}v{team_size}, replay TeamSize is {actual_team_size}")
    if any(player.get("bBot") for player in properties.get("PlayerStats", [])):
        raise ValueError("bot matches are not supported")
    if replay.get("game_type") != "TAGame.Replay_Soccar_TA":
        raise ValueError(f"unsupported game type: {replay.get('game_type')}")

    reconstructor = ReplayReconstructor(replay, sample_rate)
    worlds = reconstructor.reconstruct()
    rosters = stable_team_rosters(worlds, team_size)
    history_steps = max(1, round(history_seconds * sample_rate) + 1)
    examples: list[dict[str, Any]] = []
    track_ids: dict[tuple[str, tuple[str, ...], tuple[str, ...]], int] = {}
    track_keys: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    labels: Counter[str] = Counter()
    mechanics: Counter[str] = Counter()
    events: Counter[str] = Counter()

    for index in range(history_steps - 1, len(worlds)):
        current = worlds[index]
        history_worlds = worlds[index - history_steps + 1 : index + 1]
        if not all(world.game_active for world in history_worlds):
            continue
        future_end_frame = current.frame
        for future_index in range(index + 1, len(worlds)):
            future = worlds[future_index]
            if (
                future.time - current.time
                > target_offset_seconds + label_config.horizon_seconds + 1e-6
            ):
                break
            future_end_frame = future.frame
        if crosses_goal(
            reconstructor.goal_frames,
            worlds[index - history_steps + 1].frame,
            future_end_frame,
        ):
            continue

        for ego_id, ego in current.players.items():
            teammates = [
                player_id for player_id in rosters[ego.team] if player_id != ego_id
            ]
            opponents = rosters[1 - ego.team]
            classification = classify_target_window(
                worlds, index, ego_id, label_config, target_offset_seconds
            )
            endpoint = endpoint_target(
                worlds,
                index,
                ego_id,
                target_offset_seconds + label_config.horizon_seconds,
            )
            if classification is None or endpoint is None:
                continue
            target_position, target_forward = endpoint
            track_key = (ego_id, tuple(teammates), tuple(opponents))
            track_id = track_ids.get(track_key)
            if track_id is None:
                track_id = len(track_keys)
                track_ids[track_key] = track_id
                track_keys.append(track_key)
            examples.append(
                {
                    "replay_id": replay_id(replay, source),
                    "time": round(current.time, 4),
                    "frame": current.frame,
                    "player_id": ego_id,
                    "team": ego.team,
                    "label": classification.intent,
                    "label_confidence": round(classification.intent_confidence, 4),
                    "mechanic": classification.mechanic,
                    "mechanic_confidence": round(classification.mechanic_confidence, 4),
                    "events": classification.events,
                    "target_position": target_position,
                    "target_forward": target_forward,
                    "track_id": track_id,
                    "state_index": index,
                }
            )
            labels[classification.intent] += 1
            mechanics[classification.mechanic] += 1
            events.update(classification.events)

    track_bounds = [[len(worlds), -1] for _ in track_keys]
    for example in examples:
        bounds = track_bounds[example["track_id"]]
        bounds[0] = min(bounds[0], example["state_index"] - history_steps + 1)
        bounds[1] = max(bounds[1], example["state_index"])

    tracks: list[array] = []
    for (ego_id, teammates, opponents), (start, end) in zip(track_keys, track_bounds):
        values = array("f")
        for world in worlds[start : end + 1]:
            values.extend(
                state_features(world, ego_id, list(teammates), list(opponents), team_size)
            )
        tracks.append(values)
    for example in examples:
        example["state_index"] -= track_bounds[example["track_id"]][0]

    return examples, tracks, {
        "source": str(source),
        "replay_id": replay_id(replay, source),
        "sampled_states": len(worlds),
        "examples": len(examples),
        "labels": dict(sorted(labels.items())),
        "mechanics": dict(sorted(mechanics.items())),
        "events": dict(sorted(events.items())),
    }


def discover_inputs(paths: Iterable[str]) -> list[Path]:
    discovered: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            discovered.extend(sorted(path.rglob("*.json")))
        elif path.is_file():
            discovered.append(path)
        else:
            raise FileNotFoundError(raw_path)
    return list(dict.fromkeys(path.resolve() for path in discovered))


def metadata_path(output: Path) -> Path:
    name = output.name
    for suffix in (".sqlite", ".db"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return output.with_name(f"{name}.metadata.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="rrrocket JSON files or directories")
    parser.add_argument("-o", "--output", default="training.sqlite")
    parser.add_argument("--sample-rate", type=float, default=10.0)
    parser.add_argument("--history-seconds", type=float, default=1.0)
    parser.add_argument("--target-offset-seconds", type=float, default=0.0)
    parser.add_argument("--horizon-seconds", type=float, default=1.5)
    parser.add_argument("--team-size", type=int, default=2)
    parser.add_argument("--challenge-distance", type=float, default=350.0)
    parser.add_argument("--challenge-progress", type=float, default=300.0)
    parser.add_argument("--rotation-distance", type=float, default=400.0)
    parser.add_argument("--rotation-backtrack", type=float, default=300.0)
    parser.add_argument("--possession-distance", type=float, default=450.0)
    parser.add_argument("--hold-displacement", type=float, default=200.0)
    parser.add_argument("--airborne-height", type=float, default=100.0)
    parser.add_argument("--ball-contact-distance", type=float, default=220.0)
    parser.add_argument("--boost-pickup-gain", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.sample_rate <= 0
        or args.history_seconds < 0
        or args.target_offset_seconds < 0
        or args.horizon_seconds <= 0
    ):
        raise ValueError(
            "sample rate and horizon must be positive; history and target offset cannot be negative"
        )
    if args.team_size < 1:
        raise ValueError("team size must be at least 1")
    endpoint_steps = (args.target_offset_seconds + args.horizon_seconds) * args.sample_rate
    if not math.isclose(endpoint_steps, round(endpoint_steps), abs_tol=1e-6):
        raise ValueError("target window end must align with the configured sample rate")
    thresholds = (
        args.challenge_distance,
        args.challenge_progress,
        args.rotation_distance,
        args.rotation_backtrack,
        args.possession_distance,
        args.hold_displacement,
        args.airborne_height,
        args.ball_contact_distance,
        args.boost_pickup_gain,
    )
    if any(value < 0 for value in thresholds):
        raise ValueError("label thresholds cannot be negative")

    output = Path(args.output).resolve()
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {output.parent}")
    inputs = [path for path in discover_inputs(args.inputs) if path != output]
    config = LabelConfig(
        horizon_seconds=args.horizon_seconds,
        challenge_distance=args.challenge_distance,
        challenge_progress=args.challenge_progress,
        rotation_distance=args.rotation_distance,
        rotation_backtrack=args.rotation_backtrack,
        possession_distance=args.possession_distance,
        hold_displacement=args.hold_displacement,
        airborne_height=args.airborne_height,
        ball_contact_distance=args.ball_contact_distance,
        boost_pickup_gain=args.boost_pickup_gain,
    )
    summaries: list[dict[str, Any]] = []
    total_examples = 0
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE feature_tracks (
            id INTEGER PRIMARY KEY,
            feature_count INTEGER NOT NULL,
            state_count INTEGER NOT NULL,
            features BLOB NOT NULL
        );
        CREATE TABLE examples (
            id INTEGER PRIMARY KEY,
            replay_id TEXT NOT NULL,
            time REAL NOT NULL,
            frame INTEGER NOT NULL,
            player_id TEXT NOT NULL,
            team INTEGER NOT NULL,
            label INTEGER NOT NULL,
            label_confidence REAL NOT NULL,
            mechanic INTEGER NOT NULL,
            mechanic_confidence REAL NOT NULL,
            event_mask INTEGER NOT NULL,
            target_position_x REAL NOT NULL,
            target_position_y REAL NOT NULL,
            target_position_z REAL NOT NULL,
            target_forward_x REAL NOT NULL,
            target_forward_y REAL NOT NULL,
            target_forward_z REAL NOT NULL,
            track_id INTEGER NOT NULL REFERENCES feature_tracks(id),
            state_index INTEGER NOT NULL
        );
        """
    )
    labels = INTENT_LABELS
    mechanics = ["GROUNDED", "AERIAL", "RECOVERING"]
    event_names = ["SHOT", "CLEAR", "BOOST_PICKUP", "DEMOLITION"]
    label_ids = {name: index for index, name in enumerate(labels)}
    mechanic_ids = {name: index for index, name in enumerate(mechanics)}
    event_bits = {name: 1 << index for index, name in enumerate(event_names)}

    for source in inputs:
        try:
            with source.open(encoding="utf-8") as replay_file:
                replay = json.load(replay_file)
            if "network_frames" not in replay or "objects" not in replay:
                print(f"Skipping non-rrrocket JSON: {source}", file=sys.stderr)
                continue
            examples, tracks, summary = examples_for_replay(
                replay,
                source,
                args.sample_rate,
                args.history_seconds,
                args.team_size,
                config,
                args.target_offset_seconds,
            )
            track_ids: list[int] = []
            feature_count = len(feature_names(args.team_size))
            for track in tracks:
                if sys.byteorder != "little":
                    track.byteswap()
                cursor = connection.execute(
                    "INSERT INTO feature_tracks(feature_count, state_count, features) "
                    "VALUES (?, ?, ?)",
                    (
                        feature_count,
                        len(track) // feature_count,
                        sqlite3.Binary(track.tobytes()),
                    ),
                )
                track_ids.append(int(cursor.lastrowid))
            connection.executemany(
                """INSERT INTO examples(
                       replay_id, time, frame, player_id, team, label, label_confidence,
                       mechanic, mechanic_confidence, event_mask,
                       target_position_x, target_position_y, target_position_z,
                       target_forward_x, target_forward_y, target_forward_z,
                       track_id, state_index
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        example["replay_id"],
                        example["time"],
                        example["frame"],
                        example["player_id"],
                        example["team"],
                        label_ids[example["label"]],
                        example["label_confidence"],
                        mechanic_ids[example["mechanic"]],
                        example["mechanic_confidence"],
                        sum(event_bits[event] for event in example["events"]),
                        *example["target_position"],
                        *example["target_forward"],
                        track_ids[example["track_id"]],
                        example["state_index"],
                    )
                    for example in examples
                ],
            )
            summaries.append(summary)
            total_examples += len(examples)
            print(
                f"{source.name}: {len(examples)} examples {summary['labels']}",
                file=sys.stderr,
            )
        except (json.JSONDecodeError, OSError, TypeError, ValueError, KeyError) as error:
            print(f"Skipping {source}: {error}", file=sys.stderr)

    metadata = {
        "format_version": 6,
        "storage": "sqlite_float32_tracks",
        "float_byte_order": "little",
        "player_slot_assignment": "replay_wide_stable_player_id",
        "description": "Ego-centric Rocket League intent, mechanic, and event sequences",
        "labels": labels,
        "mechanics": mechanics,
        "events": event_names,
        "label_source": "heuristic_future_trajectory",
        "target_window": {
            "start_seconds": args.target_offset_seconds,
            "end_seconds": args.target_offset_seconds + args.horizon_seconds,
        },
        "prediction_targets": {
            "position": {
                "columns": ["target_position_x", "target_position_y", "target_position_z"],
                "coordinates": "canonical absolute field position at target_window.end_seconds",
                "scale": POSITION_SCALE,
            },
            "forward": {
                "columns": ["target_forward_x", "target_forward_y", "target_forward_z"],
                "coordinates": "canonical unit facing vector at target_window.end_seconds",
            },
        },
        "feature_names": feature_names(args.team_size),
        "sequence_length": max(1, round(args.history_seconds * args.sample_rate) + 1),
        "sample_rate_hz": args.sample_rate,
        "history_seconds": args.history_seconds,
        "team_size": args.team_size,
        "normalization": {
            "position_xyz": POSITION_SCALE,
            "velocity": VELOCITY_SCALE,
            "angular_velocity": ANGULAR_VELOCITY_SCALE,
            "rrrocket_angular_velocity_scale": RRROCKET_ANGULAR_VELOCITY_SCALE,
            "team_coordinates": "both X and Y rotated 180 degrees for Team 1; attack is +Y",
        },
        "label_config": vars(config),
        "total_examples": total_examples,
        "replays": summaries,
    }
    connection.execute(
        "INSERT INTO metadata(key, value) VALUES ('schema', ?)",
        (json.dumps(metadata, separators=(",", ":")),),
    )
    connection.commit()
    connection.close()
    temporary.replace(output)
    metadata_file = metadata_path(output)
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {total_examples} indexed examples to {output}", file=sys.stderr)
    print(f"Wrote metadata to {metadata_file}", file=sys.stderr)
    return 0 if total_examples else 1


if __name__ == "__main__":
    raise SystemExit(main())
