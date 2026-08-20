"""Checks for the bar-angle trajectory-optimization evaluator."""

import math

import numpy as np
import pytest

from dash_mjlab.trajopt import BarAngleTrajOptEnv
from dash_mjlab.trajopt.example import SPAWN, TARGET_ANGLE, make_two_phase_push


@pytest.fixture(scope="module")
def env() -> BarAngleTrajOptEnv:
  return BarAngleTrajOptEnv()


def _hold_spawn():
  return [lambda t, j=j: SPAWN[j] for j in range(4)]


def test_untouched_bar_costs_target(env: BarAngleTrajOptEnv) -> None:
  """Holding the spawn pose never touches the bar, so the cost is exactly the
  commanded angle -- gravity must not move a bar on a vertical hinge."""
  assert env.evaluate(_hold_spawn(), target_angle=0.6) == pytest.approx(0.6, abs=0.02)


def test_deterministic(env: BarAngleTrajOptEnv) -> None:
  fns = make_two_phase_push()
  assert env.evaluate(fns, TARGET_ANGLE) == env.evaluate(fns, TARGET_ANGLE)


def test_cost_wraps(env: BarAngleTrajOptEnv) -> None:
  """The cost is the shortest angular distance, never the long way round."""
  cost = env.evaluate(_hold_spawn(), target_angle=2 * math.pi - 0.3)
  assert cost == pytest.approx(0.3, abs=0.02)


def test_out_of_range_targets_are_clamped(env: BarAngleTrajOptEnv) -> None:
  fns = [lambda t: 100.0, lambda t: -100.0, lambda t: 50.0, lambda t: -50.0]
  assert math.isfinite(env.evaluate(fns, target_angle=0.5))


def test_non_finite_target_raises(env: BarAngleTrajOptEnv) -> None:
  fns = [lambda t: float("nan"), *_hold_spawn()[1:]]
  with pytest.raises(ValueError, match="non-finite"):
    env.evaluate(fns, target_angle=0.5)


def test_wrong_arity_raises(env: BarAngleTrajOptEnv) -> None:
  with pytest.raises(ValueError, match="Expected 4"):
    env.evaluate(_hold_spawn()[:3], target_angle=0.5)


def test_example_push_reaches_target(env: BarAngleTrajOptEnv) -> None:
  """The searched demo trajectory lands the bar near its target (0.005 rad
  when it was found; the loose bound is headroom for physics-engine drift)."""
  cost, traj = env.evaluate(
    make_two_phase_push(), TARGET_ANGLE, return_trajectory=True
  )
  assert cost < 0.2
  # And the bar is parked, not swinging through the target at the buzzer.
  final_speed = abs(traj["bar_angle"][-1] - traj["bar_angle"][-2]) / env.timestep
  assert final_speed < 0.5


def test_trajectory_output_shapes(env: BarAngleTrajOptEnv) -> None:
  cost, traj = env.evaluate(_hold_spawn(), 0.5, return_trajectory=True)
  n = env.num_steps
  assert traj["time"].shape == (n,)
  assert traj["bar_angle"].shape == (n,)
  assert traj["joint_pos"].shape == (n, 4)
  assert traj["joint_target"].shape == (n, 4)
  assert np.isfinite(traj["joint_pos"]).all()
