#!/usr/bin/env python3
"""Produce structured goal and player-prediction replay analysis."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent

from . import family, runtime as base  # noqa: E402
from .predictor import RollingIntentPredictor  # noqa: E402
from .reconstruction import (  # noqa: E402
    ReplayReconstructor,
    WorldState,
    canonical_vector,
    classify_target_window,
    distance,
    feature_names,
    stable_team_rosters,
    state_features,
)


WINDOW_WEIGHTS = {"0-1s": 1.0, "1-2s": 1.5, "2-3.5s": 2.25}
TEAM_GOAL_ROTATION_WEIGHT = 0.1
ROTATION_INTENTS = {"CLOSE_ROTATE", "FAR_ROTATE"}
TOUCH_DISTANCE = 240.0


@dataclass(frozen=True)
class Decision:
    time: float
    player_id: str
    player_name: str
    window: str
    expert_family: str
    expert_confidence: float
    expert_intent: str
    actual_family: str
    actual_intent: str
    actual_confidence: float
    actionable: bool
    aligned: bool = False
    segment_id: int = 0
    intent_mismatch: bool = False


@dataclass(frozen=True)
class Episode:
    player_id: str
    player_name: str
    window: str
    expert_family: str
    actual_family: str
    start_time: float
    end_time: float
    count: int
    score: float
    peak_expert_confidence: float
    peak_actual_confidence: float
    expert_intent: str
    actual_intent: str


@dataclass(frozen=True)
class Touch:
    time: float
    player_id: str
    team: int
    ball_position: tuple[float, float, float]


@dataclass(frozen=True)
class Contribution:
    time: float
    player_id: str
    player_name: str
    role: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path, help=".replay file or parsed rrrocket JSON")
    team = parser.add_mutually_exclusive_group(required=True)
    team.add_argument(
        "--team",
        type=int,
        choices=(1, 2),
        help="team to analyze: 1 Blue, 2 Orange",
    )
    team.add_argument(
        "--all-teams",
        action="store_true",
        help="analyze Blue and Orange while sharing model loading and replay reconstruction",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        help="model checkpoint; repeat for multiple target windows",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument(
        "--lookback",
        type=float,
        default=12.0,
        help="seconds of completed decisions before each goal (default: 12)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.25,
        help="seconds between decision samples (default: 0.25)",
    )
    parser.add_argument(
        "--prediction-interval",
        type=float,
        help="also emit viewer player predictions at this interval",
    )
    parser.add_argument(
        "--expert-threshold",
        type=float,
        default=0.55,
        help="minimum aggregated expert-family probability (default: 0.55)",
    )
    parser.add_argument(
        "--actual-threshold",
        type=float,
        default=0.6,
        help="minimum actual intent confidence (default: 0.6)",
    )
    parser.add_argument(
        "--minimum-persistence",
        type=int,
        default=2,
        help="samples required for a primary finding (default: 2)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=4,
        help="maximum ranked findings per goal (default: 4)",
    )
    return parser.parse_args()


def window_weight(window: str) -> float:
    return WINDOW_WEIGHTS.get(window, 1.0)


def decision_score(decision: Decision, team_goal: bool = False) -> float:
    score = (
        window_weight(decision.window)
        * decision.expert_confidence
        * decision.actual_confidence
    )
    if team_goal and decision.actual_intent in ROTATION_INTENTS:
        score *= TEAM_GOAL_ROTATION_WEIGHT
    return score


def build_episodes(
    decisions: list[Decision], interval: float, aligned: bool = False
) -> list[Episode]:
    grouped: dict[tuple[str, str, int, str, str, str, str], list[Decision]] = {}
    for decision in decisions:
        eligible = decision.aligned if aligned else decision.actionable
        if eligible:
            intents = (
                (decision.expert_intent, decision.actual_intent)
                if decision.intent_mismatch
                else ("", "")
            )
            grouped.setdefault(
                (
                    decision.player_id,
                    decision.window,
                    decision.segment_id,
                    decision.expert_family,
                    decision.actual_family,
                    *intents,
                ),
                [],
            ).append(decision)

    episodes: list[Episode] = []
    for samples in grouped.values():
        samples.sort(key=lambda item: item.time)
        run: list[Decision] = []
        for sample in samples:
            same_run = run and sample.time - run[-1].time <= interval * 2.5
            if run and not same_run:
                episodes.append(summarize_episode(run, team_goal=aligned))
                run = []
            run.append(sample)
        if run:
            episodes.append(summarize_episode(run, team_goal=aligned))
    return episodes


def summarize_episode(samples: list[Decision], team_goal: bool = False) -> Episode:
    first = samples[0]
    persistence_multiplier = 1.0 + 0.5 * (len(samples) - 1)
    score = (
        sum(decision_score(sample, team_goal=team_goal) for sample in samples)
        * persistence_multiplier
    )
    strongest = max(samples, key=lambda sample: decision_score(sample, team_goal=team_goal))
    return Episode(
        player_id=first.player_id,
        player_name=first.player_name,
        window=first.window,
        expert_family=first.expert_family,
        actual_family=first.actual_family,
        start_time=first.time,
        end_time=samples[-1].time,
        count=len(samples),
        score=score,
        peak_expert_confidence=max(item.expert_confidence for item in samples),
        peak_actual_confidence=max(item.actual_confidence for item in samples),
        expert_intent=strongest.expert_intent,
        actual_intent=strongest.actual_intent,
    )


def rank_episodes(
    decisions: list[Decision],
    interval: float,
    minimum_persistence: int,
    aligned: bool = False,
    goal_time: float | None = None,
) -> tuple[list[Episode], list[Episode]]:
    episodes = sorted(
        build_episodes(decisions, interval, aligned),
        key=lambda item: item.score
        * (
            math.exp(-max(0.0, goal_time - item.end_time) / 3.0)
            if goal_time is not None
            else 1.0
        ),
        reverse=True,
    )
    primary = [episode for episode in episodes if episode.count >= minimum_persistence]
    warnings = [episode for episode in episodes if episode.count < minimum_persistence]
    return primary, warnings


def decisions_for_goal(
    decisions: list[Decision] | deque[Decision],
    segment_id: int,
    goal_time: float,
    lookback: float,
) -> list[Decision]:
    return [
        decision
        for decision in decisions
        if decision.segment_id == segment_id
        and 0.0 <= goal_time - decision.time <= lookback
    ]


def detect_touches(worlds: list[WorldState]) -> list[Touch]:
    touches: list[Touch] = []
    contacting_player: str | None = None
    for world in worlds:
        if not world.game_active or world.ball is None:
            contacting_player = None
            continue
        candidates = [
            (distance(player.body.position, world.ball.position), player)
            for player in world.players.values()
        ]
        if not candidates:
            contacting_player = None
            continue
        contact_distance, player = min(candidates, key=lambda item: item[0])
        if contact_distance > TOUCH_DISTANCE:
            contacting_player = None
            continue
        touch = Touch(world.time, player.player_id, player.team, world.ball.position)
        if contacting_player == player.player_id and touches:
            touches[-1] = touch
        elif (
            touches
            and touches[-1].player_id == player.player_id
            and world.time - touches[-1].time <= 0.75
        ):
            touches[-1] = touch
        else:
            touches.append(touch)
        contacting_player = player.player_id
    return touches


def goal_contributions(
    worlds: list[WorldState],
    analyzed_team: int,
    roster: list[str],
    display_names: dict[str, str],
    goal_time: float,
    lookback: float,
    decisions: list[Decision] | deque[Decision] = (),
) -> list[Contribution]:
    last_inactive = max(
        (index for index, world in enumerate(worlds) if not world.game_active),
        default=-1,
    )
    segment_worlds = [
        world
        for world in worlds[last_inactive + 1 :]
        if world.game_active and 0.0 <= goal_time - world.time <= lookback
    ]
    touches = detect_touches(segment_worlds)
    team_touch_indices = [
        index
        for index, touch in enumerate(touches)
        if touch.team == analyzed_team and touch.player_id in roster
    ]
    if not team_touch_indices:
        return []

    contributions: list[Contribution] = []
    final_index = team_touch_indices[-1]
    final_touch = touches[final_index]
    if goal_time - final_touch.time > 5.0:
        return []
    final_is_team_touch = final_index == len(touches) - 1
    contributions.append(
        Contribution(
            final_touch.time,
            final_touch.player_id,
            display_names.get(final_touch.player_id, final_touch.player_id),
            "FINISH" if final_is_team_touch else "SHOT CREATION",
            (
                "made the likely final touch before the goal"
                if final_is_team_touch
                else "created the scoring ball before an opponent's final deflection"
            ),
        )
    )

    previous_teammate_index: int | None = None
    for index in range(final_index - 1, -1, -1):
        touch = touches[index]
        if touch.team != analyzed_team:
            break
        if touch.player_id != final_touch.player_id:
            previous_teammate_index = index
            break
    if (
        previous_teammate_index is not None
        and final_touch.time - touches[previous_teammate_index].time <= 5.0
    ):
        pass_touch = touches[previous_teammate_index]
        advance = (
            canonical_vector(final_touch.ball_position, analyzed_team)[1]
            - canonical_vector(pass_touch.ball_position, analyzed_team)[1]
        )
        advance_text = f" and advanced it {advance:.0f} uu" if advance >= 500.0 else ""
        contributions.append(
            Contribution(
                pass_touch.time,
                pass_touch.player_id,
                display_names.get(pass_touch.player_id, pass_touch.player_id),
                "CHANCE CREATION",
                f"played the ball to the eventual finisher{advance_text}",
            )
        )
        used_passing_option = any(
            decision.player_id == final_touch.player_id
            and decision.actual_intent == "CHERRY_PICK"
            and 0.0 <= final_touch.time - decision.time <= 3.0
            for decision in decisions
        )
        if used_passing_option:
            contributions.append(
                Contribution(
                    final_touch.time,
                    final_touch.player_id,
                    display_names.get(final_touch.player_id, final_touch.player_id),
                    "PASSING OPTION",
                    "made an upfield passing run before receiving the ball",
                )
            )

    for index in team_touch_indices:
        if index == 0:
            continue
        touch = touches[index]
        previous = touches[index - 1]
        if previous.team != analyzed_team and touch.time - previous.time <= 2.5:
            contributions.append(
                Contribution(
                    touch.time,
                    touch.player_id,
                    display_names.get(touch.player_id, touch.player_id),
                    "POSSESSION WIN",
                    "won the ball shortly after the opponent's touch",
                )
            )
            break

    pass_creator_id = (
        touches[previous_teammate_index].player_id
        if previous_teammate_index is not None
        else None
    )
    best_progress: tuple[float, Touch] | None = None
    for index in team_touch_indices:
        if index >= final_index:
            continue
        touch = touches[index]
        if touch.player_id == pass_creator_id or index + 1 >= len(touches):
            continue
        advance = (
            canonical_vector(touches[index + 1].ball_position, analyzed_team)[1]
            - canonical_vector(touch.ball_position, analyzed_team)[1]
        )
        if advance >= 800.0 and (best_progress is None or advance > best_progress[0]):
            best_progress = (advance, touch)
    if best_progress is not None:
        advance, touch = best_progress
        contributions.append(
            Contribution(
                touch.time,
                touch.player_id,
                display_names.get(touch.player_id, touch.player_id),
                "BALL PROGRESSION",
                f"moved the play {advance:.0f} uu toward the opponent's goal",
            )
        )

    role_order = {
        "POSSESSION WIN": 0,
        "BALL PROGRESSION": 1,
        "CHANCE CREATION": 2,
        "PASSING OPTION": 3,
        "SHOT CREATION": 4,
        "FINISH": 4,
    }
    return sorted(contributions, key=lambda item: (item.time, role_order[item.role]))


def overlap_summary(episodes: list[Episode]) -> str | None:
    players = {episode.player_name for episode in episodes}
    if len(players) < 2:
        return None
    engages = [episode for episode in episodes if episode.actual_family == "ENGAGE"]
    if len({episode.player_name for episode in engages}) >= 2:
        return "both teammates sustained engage behavior, indicating possible shared overcommit"
    recovers = [episode for episode in episodes if episode.actual_family == "RECOVER"]
    if len({episode.player_name for episode in recovers}) >= 2:
        return "both teammates sustained recovery behavior, indicating pressure may have been abandoned"
    return "both teammates had sustained conflicting decisions during the buildup"


def alignment_summary(episodes: list[Episode]) -> str | None:
    players = {episode.player_name for episode in episodes}
    if len(players) < 2:
        return None
    shared_families = {
        episode.expert_family
        for episode in episodes
        if any(
            item.player_name != episode.player_name
            and item.expert_family == episode.expert_family
            for item in episodes
        )
    }
    if shared_families:
        families = ", ".join(sorted(family.lower() for family in shared_families))
        return f"both teammates sustained expert-aligned {families} behavior"
    return "both teammates sustained expert-aligned decisions during the buildup"


def analyze_replay(
    replay: dict[str, Any],
    checkpoints: list[Path],
    device: str,
    analyzed_teams: list[int],
    lookback: float,
    interval: float,
    expert_threshold: float,
    actual_threshold: float,
    minimum_persistence: int,
    top: int,
    prediction_interval: float | None = None,
) -> dict[str, Any]:
    predictors = [RollingIntentPredictor(checkpoint, device) for checkpoint in checkpoints]
    predictor = predictors[0]
    sample_rate = predictor.sample_rate_hz
    if sample_rate <= 0:
        raise ValueError("checkpoint does not specify a valid sample rate")
    for candidate in predictors[1:]:
        if (
            candidate.sample_rate_hz != sample_rate
            or candidate.sequence_length != predictor.sequence_length
            or candidate.feature_names != predictor.feature_names
        ):
            raise ValueError("all checkpoints must use the same feature history schema")
    predictors.sort(key=lambda item: float(item.target_window["start_seconds"]))

    properties = replay.get("properties", {})
    team_size = properties.get("TeamSize")
    if team_size != 2:
        raise ValueError(f"this experiment requires a 2v2 replay; replay TeamSize is {team_size}")
    if feature_names(team_size) != predictor.feature_names:
        raise ValueError("checkpoint schema does not support this 2v2 replay")
    if prediction_interval is not None and any(
        player.get("bBot") for player in properties.get("PlayerStats", [])
    ):
        raise ValueError("bot matches are not supported")
    if replay.get("game_type") != "TAGame.Replay_Soccar_TA":
        raise ValueError(f"unsupported game type: {replay.get('game_type')}")

    reconstructor = ReplayReconstructor(replay, sample_rate)
    worlds = reconstructor.reconstruct()
    if not worlds:
        raise ValueError("replay contains no reconstructable states")

    rosters = stable_team_rosters(worlds, team_size)
    for analyzed_team in analyzed_teams:
        if len(rosters[analyzed_team]) != 2:
            raise ValueError("could not identify both teammates on the selected team")
    player_ids = [
        player_id
        for team in ((0, 1) if prediction_interval is not None else analyzed_teams)
        for player_id in rosters[team]
    ]
    names = base.player_names(reconstructor)
    display_names = {
        player_id: names.get(player_id, f"Player {index + 1}")
        for index, player_id in enumerate(player_ids)
    }
    inference, player_predictions = run_shared_inference(
        worlds,
        reconstructor,
        predictors,
        rosters,
        player_ids,
        display_names,
        team_size,
        interval,
        prediction_interval,
    )

    result = {
        "schemaVersion": 1,
        "teams": [
            analyze_team(
                predictors,
                reconstructor,
                worlds,
                analyzed_team,
                lookback,
                interval,
                expert_threshold,
                actual_threshold,
                minimum_persistence,
                top,
                inference,
                rosters,
                display_names,
            )
            for analyzed_team in analyzed_teams
        ],
    }
    if player_predictions is not None:
        result["playerPredictions"] = player_predictions
    return result


def run_shared_inference(
    worlds: list[WorldState],
    reconstructor: ReplayReconstructor,
    predictors: list[RollingIntentPredictor],
    rosters: dict[int, list[str]],
    player_ids: list[str],
    display_names: dict[str, str],
    team_size: int,
    goal_interval: float,
    prediction_interval: float | None,
) -> tuple[dict[int, dict[str, list[Any]]], dict[str, Any] | None]:
    predictor = predictors[0]
    teams = {
        player_id: team
        for team, roster in rosters.items()
        for player_id in roster
    }
    histories = {
        player_id: deque(maxlen=predictor.sequence_length) for player_id in player_ids
    }
    inference: dict[int, dict[str, list[Any]]] = {}
    json_players = {
        player_id: {
            "id": str(player_id),
            "displayName": display_names[player_id],
            "team": "blue" if teams[player_id] == 0 else "orange",
            "samples": [],
        }
        for player_id in player_ids
    }
    next_goal_emit = math.ceil(worlds[0].time / goal_interval) * goal_interval
    next_prediction_emit = (
        math.ceil(worlds[0].time / prediction_interval) * prediction_interval
        if prediction_interval is not None
        else None
    )
    previous_frame = worlds[0].frame

    for world_index, world in enumerate(worlds):
        scoring_teams = [
            team
            for team, goal_frames in reconstructor.team_goal_frames.items()
            if base.crossed_goal(goal_frames, previous_frame, world.frame)
        ]
        crossed_any_goal = bool(scoring_teams) or base.crossed_goal(
            reconstructor.goal_frames, previous_frame, world.frame
        )
        discontinuity = not world.game_active or crossed_any_goal
        if discontinuity:
            for history in histories.values():
                history.clear()
        else:
            for player_id in player_ids:
                ego = world.players.get(player_id)
                if ego is None:
                    histories[player_id].clear()
                    continue
                teammates = [item for item in rosters[ego.team] if item != player_id]
                histories[player_id].append(
                    state_features(
                        world,
                        player_id,
                        teammates,
                        rosters[1 - ego.team],
                        team_size,
                    )
                )
        previous_frame = world.frame

        goal_due = world.time + 1e-6 >= next_goal_emit
        if goal_due:
            while next_goal_emit <= world.time + 1e-6:
                next_goal_emit += goal_interval
        prediction_due = (
            next_prediction_emit is not None
            and world.time + 1e-6 >= next_prediction_emit
        )
        if prediction_due:
            assert prediction_interval is not None
            assert next_prediction_emit is not None
            while next_prediction_emit <= world.time + 1e-6:
                next_prediction_emit += prediction_interval
        if not goal_due and not prediction_due:
            continue
        if discontinuity:
            if prediction_due:
                for player in json_players.values():
                    player["samples"].append(base.player_sample_json(world.time))
            continue

        ready = [
            (player_id, list(histories[player_id]))
            for player_id in player_ids
            if len(histories[player_id]) == predictor.sequence_length
        ]
        batched = [
            candidate.predict_endpoint_windows([history for _, history in ready])
            for candidate in predictors
        ]
        predictions_by_player = {
            player_id: [
                batched[candidate_index][player_index]
                for candidate_index in range(len(predictors))
            ]
            for player_index, (player_id, _) in enumerate(ready)
        }
        if goal_due:
            inference[world_index] = predictions_by_player
        if prediction_due:
            for player_id, player in json_players.items():
                predictions = predictions_by_player.get(player_id)
                if predictions is None:
                    player["samples"].append(base.player_sample_json(world.time))
                    continue
                horizons = [
                    base.prediction_horizon_json(
                        candidate,
                        prediction,
                        teams[player_id],
                        world.time,
                    )
                    for candidate, prediction in zip(predictors, predictions)
                ]
                player["samples"].append(base.player_sample_json(world.time, horizons))

    player_predictions = None
    if prediction_interval is not None:
        player_predictions = {
            "sampleIntervalSeconds": float(prediction_interval),
            "players": list(json_players.values()),
        }
    return inference, player_predictions


def episode_finding(
    episode: Episode, goal_time: float, aligned: bool, sustained: bool
) -> dict[str, Any]:
    start_before = max(0.0, goal_time - episode.start_time)
    end_before = max(0.0, goal_time - episode.end_time)
    reason = (
        f"consistently followed {episode.expert_family.lower()} guidance"
        if aligned
        else (
            f"chose {episode.actual_intent.lower()} instead of "
            f"{episode.expert_intent.lower()} within {episode.expert_family.lower()} play"
            if episode.expert_family == episode.actual_family
            else family.describe_finding(episode.expert_family, episode.actual_family)
        )
    )
    expected = (
        f"{episode.expert_family} {episode.peak_expert_confidence:.0%} "
        f"({episode.expert_intent})"
    )
    observed = (
        f"{episode.actual_family} {episode.peak_actual_confidence:.0%} "
        f"({episode.actual_intent})"
    )
    return {
        "kind": "alignment" if aligned else "disagreement",
        "tone": "positive" if aligned else "negative",
        "text": (
            f"{episode.player_name}: {reason}; {episode.window}, {episode.count} samples, "
            f"{start_before:.0f}-{end_before:.0f}s before goal, score {episode.score:.2f}"
        ),
        "subject": {
            "playerId": episode.player_id,
            "displayName": episode.player_name,
        },
        "navigation": {"anchorSeconds": episode.start_time, "preRollSeconds": 3},
        "evidence": {"expected": expected, "observed": observed},
        "extensions": {
            "mistake": {
                "expectedFamily": episode.expert_family,
                "actualFamily": episode.actual_family,
                "expectedIntent": episode.expert_intent,
                "actualIntent": episode.actual_intent,
                "score": episode.score,
                "sampleCount": episode.count,
                "window": episode.window,
                "startTimeSeconds": episode.start_time,
                "endTimeSeconds": episode.end_time,
                "confidence": {
                    "expected": episode.peak_expert_confidence,
                    "actual": episode.peak_actual_confidence,
                },
                "sustained": sustained,
            }
        },
    }


def contribution_finding(
    contribution: Contribution, goal_time: float
) -> dict[str, Any]:
    seconds_before = max(0.0, goal_time - contribution.time)
    return {
        "kind": "contribution",
        "tone": "positive",
        "text": (
            f"{contribution.player_name}: {contribution.role.lower()}; "
            f"{contribution.detail}, {seconds_before:.1f}s before goal"
        ),
        "subject": {
            "playerId": contribution.player_id,
            "displayName": contribution.player_name,
        },
        "navigation": {"anchorSeconds": contribution.time, "preRollSeconds": 3},
        "evidence": {"role": contribution.role},
    }


def informational_finding(text: str, tone: str = "neutral") -> dict[str, Any]:
    return {"kind": "informational", "tone": tone, "text": text}


def goal_event(
    team_id: str,
    team_scored: bool,
    goal_number: int,
    world: WorldState,
    contributions: list[Contribution],
    selected: list[Episode],
    warnings: list[Episode],
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    if team_scored:
        if contributions:
            findings.extend(contribution_finding(item, world.time) for item in contributions)
        else:
            findings.append(informational_finding(
                "No reliable touch-chain contribution found from reconstructed states."
            ))
        if selected:
            findings.extend(
                episode_finding(item, world.time, True, True) for item in selected
            )
            summary = alignment_summary(selected)
            if summary:
                findings.append(informational_finding(f"Team pattern: {summary}", "positive"))
        elif warnings:
            findings.append(episode_finding(warnings[0], world.time, True, False))
        else:
            findings.append(informational_finding(
                "No sustained expert-aligned supporting habit found."
            ))
    elif selected:
        findings.extend(
            episode_finding(item, world.time, False, True) for item in selected
        )
        summary = overlap_summary(selected)
        if summary:
            findings.append(informational_finding(f"Team pattern: {summary}", "warning"))
    else:
        findings.append(informational_finding(
            "No sustained high-confidence cross-family cause found."
        ))
        if warnings:
            findings.append(episode_finding(warnings[0], world.time, False, False))

    relation = "scored" if team_scored else "conceded"
    for finding_number, finding in enumerate(findings, 1):
        finding["id"] = f"{team_id}-{relation}-{goal_number}-finding-{finding_number}"
    return {
        "id": f"{team_id}-{relation}-{goal_number}",
        "relation": relation,
        "ordinal": goal_number,
        "occurredAtSeconds": world.time,
        "displayClock": base.format_duration(world.time),
        "findings": findings,
    }


def analyze_team(
    predictors: list[RollingIntentPredictor],
    reconstructor: ReplayReconstructor,
    worlds: list[WorldState],
    analyzed_team: int,
    lookback: float,
    interval: float,
    expert_threshold: float,
    actual_threshold: float,
    minimum_persistence: int,
    top: int,
    inference: dict[int, dict[str, list[Any]]],
    rosters: dict[int, list[str]],
    display_names: dict[str, str],
) -> dict[str, Any]:
    roster = rosters[analyzed_team]
    recent: deque[Decision] = deque()
    segment_id = 0
    segment_start_index = 0
    previous_frame = worlds[0].frame
    conceded = 0
    scored = 0
    events: list[dict[str, Any]] = []

    team_name = base.TEAM_NAMES[analyzed_team]
    for world_index, world in enumerate(worlds):
        scoring_teams = [
            team
            for team, goal_frames in reconstructor.team_goal_frames.items()
            if base.crossed_goal(goal_frames, previous_frame, world.frame)
        ]
        crossed_any_goal = bool(scoring_teams) or base.crossed_goal(
            reconstructor.goal_frames, previous_frame, world.frame
        )
        for scoring_team in scoring_teams:
            relevant = decisions_for_goal(recent, segment_id, world.time, lookback)
            team_scored = scoring_team == analyzed_team
            if team_scored:
                scored += 1
                goal_number = scored
            else:
                conceded += 1
                goal_number = conceded
            primary, warnings = rank_episodes(
                relevant,
                interval,
                minimum_persistence,
                aligned=team_scored,
                goal_time=world.time,
            )
            selected = primary[:top]
            contributions = (
                goal_contributions(
                    worlds[segment_start_index:world_index],
                    analyzed_team,
                    roster,
                    display_names,
                    world.time,
                    lookback,
                    relevant,
                )
                if team_scored
                else []
            )
            events.append(goal_event(
                "blue" if analyzed_team == 0 else "orange",
                team_scored,
                goal_number,
                world,
                contributions,
                selected,
                warnings,
            ))

        discontinuity = not world.game_active or crossed_any_goal
        if discontinuity:
            recent.clear()
            segment_id += 1
            segment_start_index = world_index + 1
        previous_frame = world.frame

        predictions_by_player = inference.get(world_index)
        if discontinuity or predictions_by_player is None:
            continue
        cutoff = world.time - lookback
        while recent and recent[0].time < cutoff:
            recent.popleft()
        for player_id in roster:
            player_predictions = predictions_by_player.get(player_id, [])
            for candidate, prediction in zip(predictors, player_predictions):
                probabilities = prediction.probabilities
                actual = classify_target_window(
                    worlds,
                    world_index,
                    player_id,
                    base.label_config(candidate),
                    float(candidate.target_window["start_seconds"]),
                    reconstructor.goal_frames,
                    clip_at_goal=True,
                )
                if actual is None:
                    continue
                comparison = family.family_comparison(
                    probabilities,
                    actual.intent,
                    actual.intent_confidence,
                    expert_threshold,
                    actual_threshold,
                )
                expert_family, expert_intent, _, actual_family, _, actionable = comparison
                expert_confidence = family.aggregate_families(probabilities)[expert_family]
                intent_mismatch = (
                    expert_family != "UNKNOWN"
                    and actual_family == expert_family
                    and expert_intent != actual.intent
                    and expert_confidence >= expert_threshold
                    and actual.intent_confidence >= actual_threshold
                )
                actionable = actionable or intent_mismatch
                aligned = (
                    expert_family != "UNKNOWN"
                    and actual_family == expert_family
                    and expert_confidence >= expert_threshold
                    and actual.intent_confidence >= actual_threshold
                )
                recent.append(
                    Decision(
                        time=world.time,
                        player_id=player_id,
                        player_name=display_names[player_id],
                        window=base.window_name(candidate),
                        expert_family=expert_family,
                        expert_confidence=expert_confidence,
                        expert_intent=expert_intent,
                        actual_family=actual_family,
                        actual_intent=actual.intent,
                        actual_confidence=actual.intent_confidence,
                        actionable=actionable,
                        aligned=aligned,
                        segment_id=segment_id,
                        intent_mismatch=intent_mismatch,
                    )
                )

    team_id = "blue" if analyzed_team == 0 else "orange"
    return {
        "id": team_id,
        "team": {"id": team_id, "displayName": team_name},
        "players": [
            {"id": player_id, "displayName": display_names[player_id]}
            for player_id in roster
        ],
        "score": {"for": scored, "against": conceded},
        "events": events,
    }


def main() -> int:
    args = parse_args()
    if (
        args.lookback <= 0
        or args.interval <= 0
        or args.prediction_interval is not None and args.prediction_interval <= 0
    ):
        raise ValueError("lookback and intervals must be positive")
    if args.minimum_persistence < 1 or args.top < 1:
        raise ValueError("minimum persistence and top must be at least 1")
    for name, value in (
        ("expert threshold", args.expert_threshold),
        ("actual threshold", args.actual_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    replay = base.load_replay(args.replay.resolve(), sys.stderr)
    checkpoints = (
        [path.resolve() for path in args.checkpoint]
        if args.checkpoint
        else base.TACTICAL_CHECKPOINTS
        if all(path.is_file() for path in base.TACTICAL_CHECKPOINTS)
        else [base.DEFAULT_CHECKPOINT]
    )
    result = analyze_replay(
        replay,
        checkpoints,
        args.device,
        [0, 1] if args.all_teams else [args.team - 1],
        args.lookback,
        args.interval,
        args.expert_threshold,
        args.actual_threshold,
        args.minimum_persistence,
        args.top,
        args.prediction_interval,
    )
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
