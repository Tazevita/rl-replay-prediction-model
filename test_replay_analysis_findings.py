import json
import unittest

from replay_analysis_service.analysis import (
    Decision,
    Episode,
    build_episodes,
    episode_finding,
    goal_event,
)
from replay_analysis_service.reconstruction import WorldState


def decision(time: float, player_id: str = "player-123") -> Decision:
    return Decision(
        time=time,
        player_id=player_id,
        player_name="Aqua",
        window="1-2s",
        expert_family="CONTAIN",
        expert_confidence=0.8,
        expert_intent="SHADOW",
        actual_family="ENGAGE",
        actual_intent="CHALLENGE",
        actual_confidence=0.7,
        actionable=True,
    )


class ReplayAnalysisFindingTests(unittest.TestCase):
    def test_episode_retains_stable_player_id(self):
        episodes = build_episodes([decision(4.0), decision(4.25)], interval=0.25)

        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].player_id, "player-123")

    def test_finding_exposes_json_compatible_mistake_metadata(self):
        episode = Episode(
            player_id="player-123",
            player_name="Aqua",
            window="1-2s",
            expert_family="CONTAIN",
            actual_family="ENGAGE",
            start_time=4.0,
            end_time=4.25,
            count=2,
            score=1.68,
            peak_expert_confidence=0.8,
            peak_actual_confidence=0.7,
            expert_intent="SHADOW",
            actual_intent="CHALLENGE",
        )

        finding = episode_finding(episode, goal_time=10.0, aligned=False, sustained=True)

        self.assertEqual(
            finding["subject"],
            {"playerId": "player-123", "displayName": "Aqua"},
        )
        self.assertEqual(
            finding["extensions"]["mistake"],
            {
                "expectedFamily": "CONTAIN",
                "actualFamily": "ENGAGE",
                "expectedIntent": "SHADOW",
                "actualIntent": "CHALLENGE",
                "score": 1.68,
                "sampleCount": 2,
                "window": "1-2s",
                "startTimeSeconds": 4.0,
                "endTimeSeconds": 4.25,
                "confidence": {"expected": 0.8, "actual": 0.7},
                "sustained": True,
            },
        )
        self.assertEqual(
            finding["text"],
            "Aqua: likely overcommitted instead of protecting space; "
            "1-2s, 2 samples, 6-6s before goal, score 1.68",
        )
        json.dumps(finding, allow_nan=False)

    def test_warning_fallback_is_not_sustained(self):
        episode = build_episodes([decision(4.0)], interval=0.25)[0]

        world = WorldState(10.0, 100, 290.0, False, True, (0, 1), None, {})
        primary_event = goal_event(
            "blue", False, 1, world, [], [episode], []
        )
        fallback_event = goal_event(
            "blue", False, 1, world, [], [], [episode]
        )

        self.assertIs(
            primary_event["findings"][0]["extensions"]["mistake"]["sustained"],
            True,
        )
        self.assertIs(
            fallback_event["findings"][1]["extensions"]["mistake"]["sustained"],
            False,
        )


if __name__ == "__main__":
    unittest.main()
