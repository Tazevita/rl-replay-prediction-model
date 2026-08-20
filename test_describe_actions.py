import unittest

from describe_actions import (
    Gap,
    PlayerAction,
    TimelineSample,
    format_duration,
    format_sample,
    sampled_events,
)


def sample(time, label="OTHER", game_seconds=300.0):
    action = PlayerAction("p1", 0, label, 0.75)
    return TimelineSample(time, round(time * 10), game_seconds, False, {"p1": action})


class DescribeActionsTests(unittest.TestCase):
    def test_formats_elapsed_and_game_clocks(self):
        self.assertEqual(format_duration(62.34, tenths=True), "01:02.3")
        self.assertEqual(format_duration(59.96, tenths=True), "01:00.0")
        line = format_sample(sample(62.3, "CHALLENGE", 245.0), ["p1"], {"p1": "Aqua"}, {"p1": 0}, False)
        self.assertIn("01:02.3 elapsed | 04:05 game", line)
        self.assertIn("Blue Aqua: challenging", line)

    def test_samples_at_requested_interval(self):
        samples = [sample(index / 10) for index in range(31)]
        selected = sampled_events(samples, 1.0, None, None)
        self.assertEqual([event.time for event in selected], [0.0, 1.0, 2.0, 3.0])

    def test_formats_mechanic_and_events(self):
        timeline_sample = sample(1.0, "POSSESS")
        timeline_sample.actions["p1"].mechanic = "AERIAL"
        timeline_sample.actions["p1"].events = ("SHOT", "BOOST_PICKUP")
        line = format_sample(
            timeline_sample, ["p1"], {"p1": "Aqua"}, {"p1": 0}, False
        )
        self.assertIn("possessing / aerial / shot, boost pickup", line)

    def test_formats_close_and_far_rotations_separately(self):
        close_line = format_sample(
            sample(1.0, "CLOSE_ROTATE"), ["p1"], {"p1": "Aqua"}, {"p1": 0}, False
        )
        far_line = format_sample(
            sample(1.0, "FAR_ROTATE"), ["p1"], {"p1": "Aqua"}, {"p1": 0}, False
        )
        self.assertIn("rotating close to the ball", close_line)
        self.assertIn("rotating opposite the ball", far_line)

    def test_formats_cherry_pick_as_waiting_for_pass(self):
        line = format_sample(
            sample(1.0, "CHERRY_PICK"), ["p1"], {"p1": "Aqua"}, {"p1": 0}, False
        )

        self.assertIn("waiting upfield for a pass", line)

    def test_marks_inactive_gap(self):
        selected = sampled_events([sample(1.0), sample(1.1), sample(5.0)], 1.0, None, None)
        self.assertTrue(any(isinstance(event, Gap) for event in selected))


if __name__ == "__main__":
    unittest.main()
