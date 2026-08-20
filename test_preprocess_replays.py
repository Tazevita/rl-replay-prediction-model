import unittest
import math

from preprocess_replays import (
    Actor,
    BodyState,
    INTENT_LABELS,
    LabelConfig,
    PlayerState,
    WorldState,
    body_from_actor,
    canonical_vector,
    classify_action,
    classify_target_window,
    endpoint_target,
    feature_names,
    label_action,
    stable_team_rosters,
    state_features,
)


def body(x=0.0, y=0.0, z=17.0, vx=0.0, vy=0.0, vz=0.0):
    return BodyState(
        position=(x, y, z),
        velocity=(vx, vy, vz),
        angular_velocity=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )


def yaw_rotation(radians):
    return (0.0, 0.0, math.sin(radians / 2.0), math.cos(radians / 2.0))


def world(time, ego_y, ball_y, team=0):
    ego = PlayerState("ego", team, body(y=ego_y), 0.5)
    return WorldState(
        time,
        round(time * 10),
        200.0,
        False,
        True,
        (0, 0),
        body(y=ball_y),
        {"ego": ego},
    )


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.config = LabelConfig(1.5, 350.0, 300.0, 400.0, 300.0)

    def test_team_one_is_rotated_into_attacking_coordinates(self):
        self.assertEqual(canonical_vector((10.0, -20.0, 30.0), 1), (-10.0, 20.0, 30.0))

    def test_challenge_label_uses_future_approach(self):
        worlds = [world(0.0, 0.0, 1500.0), world(0.75, 700.0, 1500.0), world(1.5, 1200.0, 1500.0)]
        self.assertEqual(label_action(worlds, 0, "ego", self.config)[0], "CHALLENGE")

    def test_target_window_labels_later_action_without_moving_anchor(self):
        config = LabelConfig(1.0, 350.0, 300.0, 400.0, 300.0)
        worlds = [
            world(0.0, 0.0, 1500.0),
            world(0.5, 0.0, 1500.0),
            world(1.0, 0.0, 1500.0),
            world(1.5, 700.0, 1500.0),
            world(2.0, 1200.0, 1500.0),
        ]
        self.assertEqual(classify_target_window(worlds, 0, "ego", config).intent, "HOLD")
        self.assertEqual(
            classify_target_window(worlds, 0, "ego", config, 1.0).intent,
            "CHALLENGE",
        )

    def test_endpoint_target_uses_window_end_and_canonical_coordinates(self):
        worlds = [
            world(0.0, 0.0, 1500.0, team=1),
            world(1.0, -1200.0, 1500.0, team=1),
        ]
        worlds[1].players["ego"].body.position = (500.0, -1200.0, 17.0)
        worlds[1].players["ego"].body.rotation = yaw_rotation(math.pi / 2.0)

        position, forward = endpoint_target(worlds, 0, "ego", 1.0)

        self.assertAlmostEqual(position[0], -500.0 / 4096.0)
        self.assertAlmostEqual(position[1], 1200.0 / 5120.0)
        self.assertAlmostEqual(position[2], 17.0 / 2044.0)
        self.assertAlmostEqual(forward[0], 0.0, places=6)
        self.assertAlmostEqual(forward[1], -1.0, places=6)

    def test_endpoint_targets_all_three_model_marks(self):
        worlds = [world(step / 2.0, step * 500.0, 4000.0) for step in range(8)]

        one_second, _ = endpoint_target(worlds, 0, "ego", 1.0)
        two_seconds, _ = endpoint_target(worlds, 0, "ego", 2.0)
        three_and_half_seconds, _ = endpoint_target(worlds, 0, "ego", 3.5)

        self.assertAlmostEqual(one_second[1], 1000.0 / 5120.0)
        self.assertAlmostEqual(two_seconds[1], 2000.0 / 5120.0)
        self.assertAlmostEqual(three_and_half_seconds[1], 3500.0 / 5120.0)

    def test_target_window_does_not_cross_discontinuity(self):
        config = LabelConfig(1.0, 350.0, 300.0, 400.0, 300.0)
        worlds = [
            world(0.0, 0.0, 1500.0),
            world(0.5, 0.0, 1500.0),
            world(1.0, 0.0, 1500.0),
            world(1.5, 700.0, 1500.0),
            world(2.0, 1200.0, 1500.0),
        ]
        worlds[1].game_active = False
        self.assertIsNone(classify_target_window(worlds, 0, "ego", config, 1.0))
        worlds[1].game_active = True
        self.assertIsNone(
            classify_target_window(worlds, 0, "ego", config, 1.0, goal_frames=[5])
        )

    def test_target_window_can_clip_observation_before_goal(self):
        config = LabelConfig(1.0, 350.0, 150.0, 400.0, 300.0)
        worlds = [
            world(0.0, 0.0, 1500.0),
            world(0.25, 250.0, 1500.0),
            world(0.5, 500.0, 1500.0),
            world(0.75, 750.0, 1500.0),
            world(1.0, 1000.0, 1500.0),
        ]

        self.assertIsNone(
            classify_target_window(worlds, 0, "ego", config, goal_frames=[8])
        )
        clipped = classify_target_window(
            worlds, 0, "ego", config, goal_frames=[8], clip_at_goal=True
        )

        self.assertIsNotNone(clipped)
        self.assertEqual(clipped.intent, "OTHER")

    def test_clipped_target_window_cannot_start_after_goal(self):
        config = LabelConfig(0.5, 350.0, 150.0, 400.0, 300.0)
        worlds = [
            world(0.0, 0.0, 1500.0),
            world(0.5, 250.0, 1500.0),
            world(1.0, 500.0, 1500.0),
            world(1.5, 750.0, 1500.0),
        ]

        self.assertIsNone(
            classify_target_window(
                worlds,
                0,
                "ego",
                config,
                offset_seconds=1.0,
                goal_frames=[8],
                clip_at_goal=True,
            )
        )

    def test_close_rotation_uses_canonical_backtracking(self):
        worlds = [world(0.0, 1000.0, 3000.0), world(0.75, 600.0, 3000.0), world(1.5, 300.0, 3000.0)]
        self.assertEqual(label_action(worlds, 0, "ego", self.config)[0], "CLOSE_ROTATE")

    def test_far_rotation_finishes_opposite_the_ball(self):
        worlds = [world(0.0, 1000.0, 3000.0), world(0.75, 600.0, 3000.0), world(1.5, 300.0, 3000.0)]
        for snapshot, ego_x in zip(worlds, (500.0, 0.0, -500.0)):
            snapshot.ball.position = (1000.0, snapshot.ball.position[1], 17.0)
            snapshot.players["ego"].body.position = (
                ego_x,
                snapshot.players["ego"].body.position[1],
                17.0,
            )
        self.assertEqual(label_action(worlds, 0, "ego", self.config)[0], "FAR_ROTATE")

    def test_close_rotation_finishes_near_the_ball(self):
        worlds = [world(0.0, 1000.0, 3000.0), world(0.75, 600.0, 3000.0), world(1.5, 300.0, 3000.0)]
        for snapshot, ego_x in zip(worlds, (-500.0, 0.0, 500.0)):
            snapshot.ball.position = (1000.0, snapshot.ball.position[1], 17.0)
            snapshot.players["ego"].body.position = (
                ego_x,
                snapshot.players["ego"].body.position[1],
                17.0,
            )
        self.assertEqual(label_action(worlds, 0, "ego", self.config)[0], "CLOSE_ROTATE")

    def test_possession_requires_sustained_coupled_motion(self):
        worlds = [world(0.0, 0.0, 300.0), world(0.75, 400.0, 700.0), world(1.5, 800.0, 1100.0)]
        for snapshot in worlds:
            snapshot.players["ego"].body.velocity = (0.0, 550.0, 0.0)
            snapshot.ball.velocity = (0.0, 600.0, 0.0)
        self.assertEqual(classify_action(worlds, 0, "ego", self.config).intent, "POSSESS")

    def test_nearby_player_moving_against_ball_does_not_possess(self):
        worlds = [world(0.0, 0.0, 300.0), world(0.75, 400.0, 700.0), world(1.5, 800.0, 1100.0)]
        for snapshot in worlds:
            snapshot.players["ego"].body.velocity = (0.0, 900.0, 0.0)
            snapshot.ball.velocity = (0.0, -900.0, 0.0)
        self.assertNotEqual(classify_action(worlds, 0, "ego", self.config).intent, "POSSESS")

    def test_air_dribble_allows_touch_velocity_differences(self):
        worlds = [world(0.0, 0.0, 250.0), world(0.75, 600.0, 850.0), world(1.5, 1200.0, 1450.0)]
        for snapshot in worlds:
            snapshot.players["ego"].body.position = (
                0.0,
                snapshot.players["ego"].body.position[1],
                700.0,
            )
            snapshot.ball.position = (0.0, snapshot.ball.position[1], 900.0)
            snapshot.players["ego"].body.velocity = (0.0, 800.0, 100.0)
            snapshot.ball.velocity = (0.0, 2000.0, 400.0)
        self.assertEqual(classify_action(worlds, 0, "ego", self.config).intent, "POSSESS")

    def test_possession_can_bridge_one_touch_impulse(self):
        worlds = [
            world(0.0, 0.0, 250.0),
            world(0.5, 300.0, 550.0),
            world(1.0, 600.0, 850.0),
            world(1.5, 900.0, 1150.0),
        ]
        for snapshot in worlds:
            snapshot.players["ego"].body.velocity = (0.0, 600.0, 0.0)
            snapshot.ball.velocity = (0.0, 650.0, 0.0)
        worlds[1].ball.velocity = (0.0, 1800.0, 0.0)
        self.assertEqual(classify_action(worlds, 0, "ego", self.config).intent, "POSSESS")

    def test_stationary_car_near_stationary_ball_does_not_possess(self):
        worlds = [world(0.0, 0.0, 300.0), world(0.75, 0.0, 300.0), world(1.5, 0.0, 300.0)]
        self.assertNotEqual(classify_action(worlds, 0, "ego", self.config).intent, "POSSESS")

    def test_shadowing_tracks_incoming_opponent_possession(self):
        worlds = [world(0.0, 0.0, 1000.0), world(0.75, -300.0, 900.0), world(1.5, -600.0, 800.0)]
        worlds[0].ball.velocity = (0.0, -600.0, 0.0)
        worlds[0].players["opponent"] = PlayerState(
            "opponent", 1, body(y=1100.0, vy=-600.0), 0.5
        )
        self.assertEqual(classify_action(worlds, 0, "ego", self.config).intent, "SHADOW")

    def test_retreat_without_opponent_possession_is_not_shadowing(self):
        worlds = [world(0.0, 0.0, 1000.0), world(0.75, -300.0, 900.0), world(1.5, -600.0, 800.0)]
        worlds[0].ball.velocity = (0.0, -600.0, 0.0)
        self.assertEqual(classify_action(worlds, 0, "ego", self.config).intent, "CLOSE_ROTATE")

    def test_shadowing_predicts_a_side_wall_bounce(self):
        worlds = [world(0.0, -1500.0, 0.0), world(0.75, -1800.0, -500.0), world(1.5, -2100.0, -1000.0)]
        for snapshot in worlds:
            snapshot.players["ego"].body.position = (1800.0, snapshot.players["ego"].body.position[1], 17.0)
        worlds[0].ball = body(x=3900.0, vy=-800.0, vx=1600.0)
        worlds[0].players["opponent"] = PlayerState(
            "opponent", 1, body(x=3900.0, y=100.0, vx=1600.0, vy=-800.0), 0.5
        )
        self.assertEqual(classify_action(worlds, 0, "ego", self.config).intent, "SHADOW")

    def test_retreating_inside_the_net_is_not_shadowing(self):
        worlds = [world(0.0, -5300.0, -4000.0), world(0.75, -5450.0, -4100.0), world(1.5, -5600.0, -4200.0)]
        worlds[0].ball.velocity = (0.0, -500.0, 0.0)
        worlds[0].players["opponent"] = PlayerState(
            "opponent", 1, body(y=-3900.0, vy=-500.0), 0.5
        )
        self.assertNotEqual(classify_action(worlds, 0, "ego", self.config).intent, "SHADOW")

    def test_supporting_tracks_play_behind_closer_teammate(self):
        worlds = [world(0.0, -1000.0, 1000.0), world(0.75, -700.0, 1100.0), world(1.5, -400.0, 1200.0)]
        for snapshot in worlds:
            snapshot.players["mate"] = PlayerState("mate", 0, body(y=200.0), 0.5)
        self.assertEqual(classify_action(worlds, 0, "ego", self.config).intent, "SUPPORT")

    def test_support_allows_lateral_corner_coverage(self):
        worlds = [world(0.0, 0.0, 100.0), world(0.75, 0.0, 200.0), world(1.5, 0.0, 300.0)]
        for snapshot, ego_x, ball_x in zip(worlds, (0.0, 200.0, 400.0), (2000.0, 2100.0, 2200.0)):
            snapshot.players["ego"].body.position = (ego_x, 0.0, 17.0)
            snapshot.ball.position = (ball_x, snapshot.ball.position[1], 17.0)
            snapshot.players["mate"] = PlayerState(
                "mate", 0, body(x=ball_x - 200.0, y=-100.0), 0.5
            )
        self.assertEqual(classify_action(worlds, 0, "ego", self.config).intent, "SUPPORT")

    def test_support_allows_controlled_retreat(self):
        worlds = [world(0.0, -1000.0, 1000.0), world(0.75, -1100.0, 1000.0), world(1.5, -1175.0, 1000.0)]
        for snapshot in worlds:
            snapshot.players["mate"] = PlayerState("mate", 0, body(y=200.0), 0.5)
        self.assertEqual(classify_action(worlds, 0, "ego", self.config).intent, "SUPPORT")

    def test_support_expands_spacing_for_aerial_first_man(self):
        worlds = [world(0.0, -1000.0, 3200.0), world(0.75, -700.0, 3400.0), world(1.5, -400.0, 3600.0)]
        for snapshot in worlds:
            snapshot.ball.position = (0.0, snapshot.ball.position[1], 1100.0)
            snapshot.players["mate"] = PlayerState(
                "mate", 0, body(y=snapshot.ball.position[1] - 200.0, z=900.0), 0.5
            )
        self.assertEqual(classify_action(worlds, 0, "ego", self.config).intent, "SUPPORT")

    def test_teammate_far_from_ball_is_not_first_man(self):
        worlds = [world(0.0, -1000.0, 3000.0), world(0.75, -700.0, 3200.0), world(1.5, -400.0, 3400.0)]
        for snapshot in worlds:
            snapshot.players["mate"] = PlayerState("mate", 0, body(y=1000.0), 0.5)
        self.assertNotEqual(classify_action(worlds, 0, "ego", self.config).intent, "SUPPORT")

    def test_stationary_upfield_player_waiting_for_engaged_teammate_cherry_picks(self):
        worlds = [
            world(0.0, 2500.0, 1000.0),
            world(0.75, 2500.0, 1100.0),
            world(1.5, 2500.0, 1200.0),
        ]
        for snapshot in worlds:
            snapshot.players["mate"] = PlayerState(
                "mate", 0, body(y=snapshot.ball.position[1] - 200.0), 0.5
            )

        self.assertEqual(
            classify_action(worlds, 0, "ego", self.config).intent,
            "CHERRY_PICK",
        )

    def test_player_moving_upfield_as_passing_option_cherry_picks(self):
        worlds = [
            world(0.0, 1800.0, 600.0),
            world(0.75, 2200.0, 800.0),
            world(1.5, 2600.0, 1000.0),
        ]
        for snapshot in worlds:
            snapshot.players["mate"] = PlayerState(
                "mate", 0, body(y=snapshot.ball.position[1] - 200.0), 0.5
            )

        self.assertEqual(
            classify_action(worlds, 0, "ego", self.config).intent,
            "CHERRY_PICK",
        )

    def test_upfield_player_is_not_cherry_picking_without_engaged_teammate(self):
        worlds = [
            world(0.0, 2500.0, 1000.0),
            world(0.75, 2500.0, 1100.0),
            world(1.5, 2500.0, 1200.0),
        ]
        for snapshot in worlds:
            snapshot.players["mate"] = PlayerState("mate", 0, body(y=-2000.0), 0.5)

        self.assertNotEqual(
            classify_action(worlds, 0, "ego", self.config).intent,
            "CHERRY_PICK",
        )

    def test_upfield_player_is_not_cherry_picking_during_opponent_possession(self):
        worlds = [
            world(0.0, 2500.0, 1000.0),
            world(0.75, 2500.0, 1100.0),
            world(1.5, 2500.0, 1200.0),
        ]
        for snapshot in worlds:
            snapshot.players["mate"] = PlayerState(
                "mate", 0, body(y=snapshot.ball.position[1] - 200.0), 0.5
            )
        worlds[0].players["opponent"] = PlayerState(
            "opponent", 1, body(y=1050.0), 0.5
        )

        self.assertNotEqual(
            classify_action(worlds, 0, "ego", self.config).intent,
            "CHERRY_PICK",
        )

    def test_boost_detour_takes_precedence_over_cherry_pick(self):
        worlds = [
            world(0.0, 2500.0, 1500.0),
            world(0.75, 3300.0, 1700.0),
            world(1.5, 4096.0, 2000.0),
        ]
        for snapshot, ego_x in zip(worlds, (2000.0, 2600.0, 3072.0)):
            snapshot.players["ego"].body.position = (
                ego_x,
                snapshot.players["ego"].body.position[1],
                17.0,
            )
            snapshot.players["mate"] = PlayerState(
                "mate", 0, body(y=snapshot.ball.position[1] - 200.0), 0.5
            )

        self.assertEqual(
            classify_action(worlds, 0, "ego", self.config).intent,
            "BOOST_DETOUR",
        )

    def test_stationary_player_holds(self):
        worlds = [world(0.0, 2000.0, 4000.0), world(0.75, 2000.0, 4000.0), world(1.5, 2000.0, 4000.0)]
        self.assertEqual(classify_action(worlds, 0, "ego", self.config).intent, "HOLD")

    def test_boost_detour_requires_leaving_the_play_for_a_large_pad(self):
        worlds = [
            world(0.0, 3000.0, -1000.0),
            world(0.75, 3500.0, -1000.0),
            world(1.5, 4000.0, -1000.0),
        ]
        for snapshot, ego_x in zip(worlds, (2000.0, 2500.0, 3000.0)):
            snapshot.players["ego"].body.position = (ego_x, snapshot.players["ego"].body.position[1], 17.0)
        self.assertEqual(classify_action(worlds, 0, "ego", self.config).intent, "BOOST_DETOUR")

    def test_rotation_takes_precedence_over_boost_detour(self):
        worlds = [
            world(0.0, -3000.0, 1000.0),
            world(0.75, -3600.0, 1000.0),
            world(1.5, -4096.0, 1000.0),
        ]
        for snapshot in worlds:
            snapshot.players["ego"].body.position = (
                3072.0,
                snapshot.players["ego"].body.position[1],
                17.0,
            )
        self.assertEqual(classify_action(worlds, 0, "ego", self.config).intent, "CLOSE_ROTATE")

    def test_bump_pursues_an_opponent_away_from_the_ball(self):
        worlds = [world(0.0, 0.0, 3000.0), world(0.75, 0.0, 3000.0), world(1.5, 0.0, 3000.0)]
        for snapshot, ego_x, opponent_x in zip(
            worlds,
            (0.0, 500.0, 900.0),
            (1000.0, 1050.0, 1100.0),
        ):
            snapshot.players["ego"].body = body(x=ego_x, vx=900.0)
            snapshot.players["opponent"] = PlayerState(
                "opponent", 1, body(x=opponent_x), 0.5
            )
        self.assertEqual(classify_action(worlds, 0, "ego", self.config).intent, "BUMP")

    def test_bumped_player_is_not_credited_with_a_bump(self):
        worlds = [world(0.0, 0.0, 3000.0), world(0.75, 0.0, 3000.0), world(1.5, 0.0, 3000.0)]
        for snapshot, ego_x, opponent_x in zip(
            worlds,
            (0.0, 100.0, 200.0),
            (1000.0, 450.0, 250.0),
        ):
            snapshot.players["ego"].body.position = (ego_x, 0.0, 17.0)
            snapshot.players["ego"].body.rotation = yaw_rotation(math.pi)
            snapshot.players["opponent"] = PlayerState(
                "opponent", 1, body(x=opponent_x), 0.5
            )
        self.assertNotEqual(classify_action(worlds, 0, "ego", self.config).intent, "BUMP")

    def test_deliberate_near_ball_bump_targets_the_opponent(self):
        worlds = [world(0.0, 0.0, 500.0), world(0.75, 0.0, 500.0), world(1.5, 0.0, 500.0)]
        for snapshot, ego_x, opponent_x in zip(
            worlds,
            (0.0, 500.0, 900.0),
            (1000.0, 1050.0, 1100.0),
        ):
            snapshot.players["ego"].body = body(x=ego_x, vx=900.0)
            snapshot.ball.position = (1100.0, 500.0, 17.0)
            snapshot.players["opponent"] = PlayerState(
                "opponent", 1, body(x=opponent_x), 0.5
            )
        self.assertEqual(classify_action(worlds, 0, "ego", self.config).intent, "BUMP")

    def test_ambiguous_near_ball_collision_remains_a_challenge(self):
        worlds = [world(0.0, 0.0, 0.0), world(0.75, 0.0, 0.0), world(1.5, 0.0, 0.0)]
        for snapshot, ego_x, opponent_x in zip(
            worlds,
            (0.0, 500.0, 900.0),
            (1200.0, 1150.0, 1100.0),
        ):
            snapshot.players["ego"].body = body(x=ego_x, vx=900.0)
            snapshot.ball.position = (1100.0, 0.0, 17.0)
            snapshot.players["opponent"] = PlayerState(
                "opponent", 1, body(x=opponent_x, vx=-100.0), 0.5
            )
        self.assertEqual(
            classify_action(worlds, 0, "ego", self.config).intent,
            "CHALLENGE",
        )

    def test_near_miss_is_not_a_bump(self):
        worlds = [world(0.0, 0.0, 3000.0), world(0.75, 0.0, 3000.0), world(1.5, 0.0, 3000.0)]
        for snapshot, ego_x, opponent_x in zip(
            worlds,
            (0.0, 500.0, 900.0),
            (1200.0, 1200.0, 1200.0),
        ):
            snapshot.players["ego"].body = body(x=ego_x, vx=900.0)
            snapshot.players["opponent"] = PlayerState(
                "opponent", 1, body(x=opponent_x), 0.5
            )
        self.assertNotEqual(classify_action(worlds, 0, "ego", self.config).intent, "BUMP")

    def test_rotation_overrides_contact_with_an_opponent(self):
        worlds = [
            world(0.0, 1000.0, 3000.0),
            world(0.75, 600.0, 3000.0),
            world(1.5, 300.0, 3000.0),
        ]
        for snapshot, ego_y, opponent_y in zip(
            worlds,
            (1000.0, 600.0, 300.0),
            (200.0, 350.0, 250.0),
        ):
            snapshot.players["ego"].body.rotation = yaw_rotation(-math.pi / 2.0)
            snapshot.players["ego"].body.velocity = (0.0, -800.0, 0.0)
            snapshot.players["opponent"] = PlayerState(
                "opponent", 1, body(y=opponent_y), 0.5
            )
        self.assertEqual(
            classify_action(worlds, 0, "ego", self.config).intent,
            "CLOSE_ROTATE",
        )

    def test_pressure_closes_on_ball_carrier_without_challenging(self):
        worlds = [world(0.0, 0.0, 1050.0), world(0.75, 250.0, 1050.0), world(1.5, 500.0, 1050.0)]
        for snapshot in worlds:
            snapshot.players["opponent"] = PlayerState(
                "opponent", 1, body(y=1000.0), 0.5
            )
        self.assertEqual(classify_action(worlds, 0, "ego", self.config).intent, "PRESSURE")

    def test_lateral_adjustment_is_repositioning(self):
        worlds = [world(0.0, 0.0, 3000.0), world(0.75, 0.0, 3000.0), world(1.5, 0.0, 3000.0)]
        for snapshot, ego_x in zip(worlds, (0.0, 500.0, 900.0)):
            snapshot.players["ego"].body.position = (ego_x, 0.0, 17.0)
        self.assertEqual(classify_action(worlds, 0, "ego", self.config).intent, "REPOSITION")

    def test_wide_defensive_return_is_rotation_with_little_net_backtracking(self):
        worlds = [
            world(0.0, -1000.0, 2000.0),
            world(0.75, -1150.0, 2000.0),
            world(1.5, -1200.0, 2000.0),
        ]
        for snapshot, ego_x in zip(worlds, (2000.0, 1000.0, 200.0)):
            snapshot.players["ego"].body.position = (
                ego_x,
                snapshot.players["ego"].body.position[1],
                17.0,
            )
            snapshot.ball.position = (2000.0, snapshot.ball.position[1], 17.0)
        self.assertEqual(
            classify_action(worlds, 0, "ego", self.config).intent,
            "CLOSE_ROTATE",
        )

    def test_moving_into_the_defensive_lane_is_defending(self):
        worlds = [
            world(0.0, -3800.0, -1000.0),
            world(0.75, -3800.0, -1000.0),
            world(1.5, -3800.0, -1000.0),
        ]
        for snapshot, ego_x in zip(worlds, (2200.0, 1300.0, 500.0)):
            snapshot.players["ego"].body.position = (ego_x, -3800.0, 17.0)
        self.assertEqual(classify_action(worlds, 0, "ego", self.config).intent, "DEFEND")

    def test_reposition_uses_full_path_instead_of_endpoint_only(self):
        worlds = [
            world(0.0, 0.0, 3000.0),
            world(0.75, 1000.0, 3000.0),
            world(1.5, 0.0, 3000.0),
        ]
        worlds[-1].players["ego"].body.position = (400.0, 0.0, 17.0)
        self.assertNotEqual(
            classify_action(worlds, 0, "ego", self.config).intent,
            "REPOSITION",
        )

    def test_committed_ball_facing_approach_is_a_challenge_setup(self):
        worlds = [
            world(0.0, 0.0, 3000.0),
            world(0.75, 500.0, 3000.0),
            world(1.5, 1100.0, 3000.0),
        ]
        for snapshot in worlds:
            snapshot.players["ego"].body.rotation = yaw_rotation(math.pi / 2.0)
        self.assertGreater(
            min(
                abs(snapshot.ball.position[1] - snapshot.players["ego"].body.position[1])
                for snapshot in worlds
            ),
            self.config.challenge_distance,
        )
        self.assertEqual(
            classify_action(worlds, 0, "ego", self.config).intent,
            "CHALLENGE",
        )

    def test_pursuing_an_advancing_loose_ball_is_attacking(self):
        worlds = [
            world(0.0, 0.0, 1500.0),
            world(0.75, 600.0, 2100.0),
            world(1.5, 1200.0, 2700.0),
        ]
        for snapshot in worlds:
            snapshot.players["ego"].body.rotation = yaw_rotation(math.pi / 2.0)
            snapshot.players["ego"].body.velocity = (0.0, 800.0, 0.0)
            snapshot.ball.velocity = (0.0, 800.0, 0.0)
        self.assertEqual(
            classify_action(worlds, 0, "ego", self.config).intent,
            "ATTACK",
        )

    def test_forward_movement_without_an_advancing_ball_is_not_attacking(self):
        worlds = [
            world(0.0, 0.0, 2500.0),
            world(0.75, 500.0, 2500.0),
            world(1.5, 900.0, 2500.0),
        ]
        for snapshot in worlds:
            snapshot.players["ego"].body.rotation = yaw_rotation(math.pi / 2.0)
        self.assertNotEqual(
            classify_action(worlds, 0, "ego", self.config).intent,
            "ATTACK",
        )

    def test_holding_the_own_goal_area_is_defending(self):
        worlds = [
            world(0.0, -4700.0, 1000.0),
            world(0.75, -4650.0, 1000.0),
            world(1.5, -4600.0, 1000.0),
        ]
        self.assertEqual(classify_action(worlds, 0, "ego", self.config).intent, "DEFEND")

    def test_new_intents_are_appended_to_preserve_existing_label_ids(self):
        self.assertEqual(
            INTENT_LABELS[:7],
            ["CHALLENGE", "POSSESS", "SUPPORT", "SHADOW", "CLOSE_ROTATE", "HOLD", "OTHER"],
        )
        self.assertEqual(INTENT_LABELS[-3:], ["FAR_ROTATE", "ATTACK", "CHERRY_PICK"])

    def test_mechanic_distinguishes_aerial_and_recovery(self):
        aerial = [world(0.0, 0.0, 2000.0), world(0.75, 100.0, 2000.0), world(1.5, 200.0, 2000.0)]
        for snapshot in aerial:
            snapshot.players["ego"].body.position = (0.0, snapshot.players["ego"].body.position[1], 500.0)
        self.assertEqual(classify_action(aerial, 0, "ego", self.config).mechanic, "AERIAL")

        aerial[0].players["ego"].body.rotation = (1.0, 0.0, 0.0, 0.0)
        self.assertEqual(classify_action(aerial, 0, "ego", self.config).mechanic, "RECOVERING")

    def test_shot_and_boost_pickup_events(self):
        worlds = [world(0.0, 0.0, 100.0), world(0.75, 300.0, 500.0), world(1.5, 800.0, 1000.0)]
        worlds[-1].players["ego"].boost = 0.8
        result = classify_action(worlds, 0, "ego", self.config)
        self.assertIn("SHOT", result.events)
        self.assertIn("BOOST_PICKUP", result.events)

    def test_nearby_disappearing_opponent_is_demolition_event(self):
        worlds = [world(0.0, 0.0, 2000.0), world(0.75, 100.0, 2000.0), world(1.5, 200.0, 2000.0)]
        worlds[0].players["opponent"] = PlayerState("opponent", 1, body(y=200.0), 0.5)
        worlds[1].players["opponent"] = PlayerState("opponent", 1, body(y=250.0), 0.5)
        self.assertIn("DEMOLITION", classify_action(worlds, 0, "ego", self.config).events)

    def test_incomplete_future_window_has_no_label(self):
        worlds = [world(0.0, 0.0, 1500.0), world(0.5, 400.0, 1500.0)]
        self.assertIsNone(label_action(worlds, 0, "ego", self.config))

    def test_inactive_future_window_has_no_label(self):
        worlds = [world(0.0, 0.0, 1500.0), world(0.75, 700.0, 1500.0), world(1.5, 1200.0, 1500.0)]
        worlds[1].game_active = False
        self.assertIsNone(label_action(worlds, 0, "ego", self.config))

    def test_feature_vector_matches_manifest(self):
        snapshot = world(0.0, 0.0, 1000.0)
        values = state_features(snapshot, "ego", [], [], 2)
        self.assertEqual(len(values), len(feature_names(2)))
        self.assertEqual(len(values), 94)

    def test_rrrocket_angular_velocity_is_converted_before_normalization(self):
        actor = Actor(
            "car",
            "TAGame.Car_TA",
            properties={
                "TAGame.RBActor_TA:ReplicatedRBState": {
                    "location": {"x": 0.0, "y": 0.0, "z": 17.0},
                    "angular_velocity": {"x": 550.0, "y": 0.0, "z": 0.0},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                }
            },
        )
        converted = body_from_actor(actor)
        self.assertIsNotNone(converted)
        snapshot = world(0.0, 0.0, 1000.0)
        snapshot.players["ego"].body = converted
        names = feature_names(2)
        values = state_features(snapshot, "ego", [], [], 2)
        self.assertAlmostEqual(values[names.index("ego_angular_vx")], 1.0)

    def test_missing_opponent_keeps_stable_slot(self):
        before = world(0.0, 0.0, 1000.0)
        missing = world(0.1, 0.0, 1000.0)
        returned = world(0.2, 0.0, 1000.0)
        for snapshot in (before, returned):
            snapshot.players["opponent_a"] = PlayerState(
                "opponent_a", 1, body(y=1500.0), 0.5
            )
            snapshot.players["opponent_b"] = PlayerState(
                "opponent_b", 1, body(y=2500.0), 0.5
            )
        missing.players["opponent_b"] = PlayerState(
            "opponent_b", 1, body(y=2500.0), 0.5
        )

        rosters = stable_team_rosters([before, missing, returned], 2)
        self.assertEqual(rosters[1], ["opponent_a", "opponent_b"])
        names = feature_names(2)
        values = state_features(missing, "ego", [], rosters[1], 2)
        self.assertEqual(values[names.index("opponent_0_valid")], 0.0)
        self.assertEqual(values[names.index("opponent_1_valid")], 1.0)
        self.assertAlmostEqual(
            values[names.index("opponent_1_abs_y")], 2500.0 / 5120.0
        )

        returned_values = state_features(returned, "ego", [], rosters[1], 2)
        self.assertEqual(returned_values[names.index("opponent_0_valid")], 1.0)
        self.assertAlmostEqual(
            returned_values[names.index("opponent_0_abs_y")], 1500.0 / 5120.0
        )

    def test_three_opponents_keep_stable_slots(self):
        complete = world(0.0, 0.0, 1000.0)
        missing = world(0.1, 0.0, 1000.0)
        for index, player_id in enumerate(("opponent_a", "opponent_b", "opponent_c")):
            complete.players[player_id] = PlayerState(
                player_id, 1, body(y=1000.0 + index * 500.0), 0.5
            )
            if player_id != "opponent_b":
                missing.players[player_id] = complete.players[player_id]

        rosters = stable_team_rosters([complete, missing], 3)
        names = feature_names(3)
        values = state_features(missing, "ego", [], rosters[1], 3)
        self.assertEqual(values[names.index("opponent_0_valid")], 1.0)
        self.assertEqual(values[names.index("opponent_1_valid")], 0.0)
        self.assertEqual(values[names.index("opponent_2_valid")], 1.0)
        self.assertAlmostEqual(
            values[names.index("opponent_2_abs_y")], 2000.0 / 5120.0
        )

    def test_rejects_more_unique_players_than_slots(self):
        snapshot = world(0.0, 0.0, 1000.0)
        for player_id in ("opponent_a", "opponent_b", "opponent_c"):
            snapshot.players[player_id] = PlayerState(player_id, 1, body(), 0.5)
        with self.assertRaisesRegex(ValueError, "only 2 stable slots"):
            stable_team_rosters([snapshot], 2)


if __name__ == "__main__":
    unittest.main()
