"""Checks for the bar-angle trajectory-optimization evaluator."""

import math

import numpy as np
import pytest

from dash_mjlab.trajopt import BarAngleTrajOptEnv
from dash_mjlab.trajopt.example import (
  TARGET_ANGLE,
  hold_spawn,
  two_phase_push,
)


@pytest.fixture(scope="module")
def env() -> BarAngleTrajOptEnv:
  return BarAngleTrajOptEnv()


@pytest.fixture(scope="module")
def env_no_penalty() -> BarAngleTrajOptEnv:
  return BarAngleTrajOptEnv(reach_penalty_weight=0.0)


def test_untouched_wheel_costs_target(env_no_penalty: BarAngleTrajOptEnv) -> None:
  """Holding the spawn pose never touches the wheel, so with the reaching term
  off the cost is exactly the commanded angle -- gravity must not move a wheel
  on a vertical hinge, and the spawn pose must be contact-free."""
  cost, traj = env_no_penalty.evaluate(
    hold_spawn, target_angle=0.6, return_trajectory=True
  )
  assert not traj["touched"]
  assert cost == pytest.approx(0.6, abs=0.02)


def test_reach_penalty_charges_closest_approach(env: BarAngleTrajOptEnv) -> None:
  """A never-touching rollout costs the angle error plus (weight times) how
  close the end-effector came to the wheel, so the flat plateau gains slope."""
  cost, traj = env.evaluate(hold_spawn, target_angle=0.6, return_trajectory=True)
  assert not traj["touched"]
  min_dist = float(traj["hand_to_wheel"].min())
  assert min_dist > 0.0
  assert cost == pytest.approx(0.6 + env.reach_penalty_weight * min_dist, abs=0.02)


def test_touching_waives_reach_penalty(env: BarAngleTrajOptEnv) -> None:
  """Once the end-effector contacts the wheel, the cost is purely the angle
  error: the demo push touches, so its cost carries no distance term."""
  cost, traj = env.evaluate(two_phase_push, TARGET_ANGLE, return_trajectory=True)
  assert traj["touched"]
  final_angle_error = abs(TARGET_ANGLE - traj["bar_angle"][-1])
  assert cost == pytest.approx(final_angle_error, abs=1e-9)


def test_single_bar_still_available() -> None:
  """num_spokes=1 recovers the original lone-bar environment."""
  env1 = BarAngleTrajOptEnv(num_spokes=1, reach_penalty_weight=0.0)
  assert env1.evaluate(hold_spawn, target_angle=0.6) == pytest.approx(0.6, abs=0.02)


def test_deterministic(env: BarAngleTrajOptEnv) -> None:
  assert env.evaluate(two_phase_push, TARGET_ANGLE) == env.evaluate(
    two_phase_push, TARGET_ANGLE
  )


def test_precompute_matches_stepwise(env: BarAngleTrajOptEnv) -> None:
  """A law that is a function of time alone cannot tell the two query modes
  apart, so the costs must agree bit for bit."""
  assert env.evaluate(two_phase_push, TARGET_ANGLE, precompute=True) == env.evaluate(
    two_phase_push, TARGET_ANGLE, precompute=False
  )


def test_cost_wraps(env_no_penalty: BarAngleTrajOptEnv) -> None:
  """The cost is the shortest angular distance, never the long way round."""
  cost = env_no_penalty.evaluate(hold_spawn, target_angle=2 * math.pi - 0.3)
  assert cost == pytest.approx(0.3, abs=0.02)


def test_out_of_range_targets_are_clamped(env: BarAngleTrajOptEnv) -> None:
  def far_outside(t: np.ndarray) -> np.ndarray:
    return np.broadcast_to([100.0, -100.0, 50.0, -50.0], (*t.shape[:-1], 4))

  assert math.isfinite(env.evaluate(far_outside, target_angle=0.5))


@pytest.mark.parametrize("precompute", [True, False])
def test_non_finite_target_raises(env: BarAngleTrajOptEnv, precompute: bool) -> None:
  def nan_pitch(t: np.ndarray) -> np.ndarray:
    return np.broadcast_to([float("nan"), 0.0, 0.0, -0.4], (*t.shape[:-1], 4))

  with pytest.raises(ValueError, match="non-finite"):
    env.evaluate(nan_pitch, target_angle=0.5, precompute=precompute)


def test_wrong_output_shape_raises(env: BarAngleTrajOptEnv) -> None:
  def three_joints(t: np.ndarray) -> np.ndarray:
    return np.broadcast_to([-0.3, 0.0, 0.0], (*t.shape[:-1], 3))

  with pytest.raises(ValueError, match="must return shape"):
    env.evaluate(three_joints, target_angle=0.5)


def test_unvectorized_law_raises(env: BarAngleTrajOptEnv) -> None:
  """A law that ignores the batch axis returns the wrong shape rather than
  silently driving every step from one sample."""
  with pytest.raises(ValueError, match="must return shape"):
    env.evaluate(lambda t: np.array([-0.3, 0.0, 0.0, -0.4]), target_angle=0.5)


def test_example_push_reaches_target(env: BarAngleTrajOptEnv) -> None:
  """The searched demo trajectory lands the wheel near its target (0.001 rad
  when it was found; the loose bound is headroom for physics-engine drift)."""
  cost, traj = env.evaluate(two_phase_push, TARGET_ANGLE, return_trajectory=True)
  assert cost < 0.2
  # And the wheel is parked, not swinging through the target at the buzzer.
  final_speed = abs(traj["bar_angle"][-1] - traj["bar_angle"][-2]) / env.timestep
  assert final_speed < 0.5


def test_trajectory_output_shapes(env: BarAngleTrajOptEnv) -> None:
  cost, traj = env.evaluate(hold_spawn, 0.5, return_trajectory=True)
  n = env.num_steps
  assert traj["time"].shape == (n,)
  assert traj["bar_angle"].shape == (n,)
  assert traj["joint_pos"].shape == (n, 4)
  assert traj["joint_target"].shape == (n, 4)
  assert traj["hand_to_wheel"].shape == (n,)
  assert traj["touched"].shape == ()
  assert np.isfinite(traj["joint_pos"]).all()
  assert np.isfinite(traj["hand_to_wheel"]).all()


def test_both_arms_mode() -> None:
  """arms="both" drives eight joints (right columns first), demands the wider
  output shape, and its do-nothing rollout matches the right-arm one closely --
  the only change at spawn is the left arm's hold gains."""
  env = BarAngleTrajOptEnv(arms="both")
  assert len(env.active_joints) == 8
  assert env.active_joints[:4] == ("r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow_pitch")
  assert all(n.startswith("l_") for n in env.active_joints[4:])

  with pytest.raises(ValueError):
    env.evaluate(hold_spawn, 0.5)  # 4 columns into an 8-joint env

  def hold8(s):
    return np.zeros((*s.shape[:-1], 8))

  cost, traj = env.evaluate(hold8, 0.5, return_trajectory=True)
  assert traj["joint_pos"].shape == (env.num_steps, 8)
  assert not bool(traj["touched"])
  right_cost = BarAngleTrajOptEnv().evaluate(hold_spawn, 0.5)
  assert abs(cost - right_cost) < 0.01
