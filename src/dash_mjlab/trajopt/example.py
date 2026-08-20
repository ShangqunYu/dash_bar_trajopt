"""Example use of :class:`BarAngleTrajOptEnv`.

Run from the repo root::

  uv run python -m dash_mjlab.trajopt.example            # print costs
  uv run python -m dash_mjlab.trajopt.example --render   # watch the push

The interface a trajectory-optimization method has to satisfy is just: four
callables, time in seconds -> desired joint position in rad, one per joint of
the right arm (shoulder pitch, shoulder roll, shoulder yaw, elbow pitch).

The push shown here was found by exactly the kind of search this environment
exists to serve -- random search over a two-phase smoothstep parameterization,
refined around the best sample. It reaches in past the bar (phase A), then
sweeps outward carrying the bar with it (phase B), and scores 0.005 rad
against a target of -0.5, where doing nothing scores 0.5.
"""

import argparse

import numpy as np

from dash_mjlab.trajopt import BarAngleTrajOptEnv

SPAWN = np.array([-0.3, 0.0, 0.0, -0.4])

TARGET_ANGLE = -0.5
# Insert: reach in past the bar, elbow straight, yaw swept inward.
POSE_A = np.array([-0.6, 0.3, 0.8, 0.0])
# Sweep: pull back out, rolling and yawing outward -- the bar rides along.
POSE_B = np.array([-0.219, -0.589, -0.005, -1.4])
T_A, D_A = 0.71, 0.52  # phase A start time and duration (s)
T_B, D_B = 2.42, 1.77  # phase B start time and duration (s)


def _smoothstep(a: float) -> float:
  a = min(max(a, 0.0), 1.0)
  return 3 * a**2 - 2 * a**3


def make_two_phase_push():
  """The four joint trajectories: spawn -> POSE_A -> POSE_B, smoothstepped."""

  def fn(j: int):
    def f(t: float) -> float:
      a1 = _smoothstep((t - T_A) / D_A)
      a2 = _smoothstep((t - T_B) / D_B)
      p = SPAWN[j] + a1 * (POSE_A[j] - SPAWN[j])
      return p + a2 * (POSE_B[j] - POSE_A[j])

    return f

  return [fn(j) for j in range(4)]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--render", action="store_true", help="Watch the rollout in a viewer."
  )
  args = parser.parse_args()

  env = BarAngleTrajOptEnv()

  hold = [lambda t, j=j: SPAWN[j] for j in range(4)]
  print(f"target angle: {TARGET_ANGLE} rad")
  print(f"hold-spawn-pose cost: {env.evaluate(hold, TARGET_ANGLE):.4f} rad")

  cost = env.evaluate(make_two_phase_push(), TARGET_ANGLE, render=args.render)
  print(f"two-phase push cost:  {cost:.4f} rad")


if __name__ == "__main__":
  main()
