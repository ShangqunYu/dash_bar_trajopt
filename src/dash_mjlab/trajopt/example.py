"""Example use of :class:`BarAngleTrajOptEnv`.

Run from the repo root::

  uv run python -m dash_mjlab.trajopt.example            # print costs
  uv run python -m dash_mjlab.trajopt.example --render   # watch the push

The interface a trajectory-optimization method has to satisfy is just: one
vectorized callable, phase in [0, 1) (shape ``(..., 1)``) -> normalized joint
targets in [-1, 1] (shape ``(..., 4)``), the columns being the right arm's
shoulder pitch, shoulder roll, shoulder yaw and elbow pitch. Zero is the spawn
pose, so no radians and no joint limits ever reach the optimizer.

The push shown here was found by exactly the kind of search this environment
exists to serve -- random search over a two-phase smoothstep family, refined
around the best sample. It hooks the wheel early and flicks it onto the
target, scoring 0.0003 rad against a target of -0.5.
Doing nothing scores 0.5 plus the reaching penalty (the closest-approach
distance to the wheel, ~0.14 m), since a rollout that never touches the wheel
is charged for how far it stayed from it.
"""

import argparse

import numpy as np
from jaxtyping import Float

from dash_mjlab.trajopt import BarAngleTrajOptEnv

TARGET_ANGLE = -0.5
# Searched on the 3-spoke wheel (random search over two-phase smoothsteps,
# refined): both legs fire in the first fifth of the horizon -- a quick hook
# with the yaw and elbow pinned at their limits, then a shoulder-pitch flick
# that spins the wheel onto the target, where the hinge damping parks it.
POSE_A = np.array([-0.4948160, 0.8244294, 1.0, 1.0])
POSE_B = np.array([0.9938336, 0.3566064, 1.0, 1.0])
S_A, W_A = 0.0260739, 0.0912601  # leg A start and duration, phase fractions
S_B, W_B = 0.1304346, 0.0100000  # leg B start and duration


def _smoothstep(a: Float[np.ndarray, "..."]) -> Float[np.ndarray, "..."]:
  a = np.clip(a, 0.0, 1.0)
  return 3 * a**2 - 2 * a**3


def two_phase_push(s: Float[np.ndarray, "... 1"]) -> Float[np.ndarray, "... 4"]:
  """Spawn -> POSE_A -> POSE_B, smoothstepped. The trailing axis broadcasts
  the phase against the four joints."""
  a1 = _smoothstep((s - S_A) / W_A)
  a2 = _smoothstep((s - S_B) / W_B)
  return a1 * POSE_A + a2 * (POSE_B - POSE_A)


def hold_spawn(s: Float[np.ndarray, "... 1"]) -> Float[np.ndarray, "... 4"]:
  """The do-nothing baseline: the zero function already is the spawn pose."""
  return np.zeros((*s.shape[:-1], 4))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--render",
    action="store_true",
    help="Watch the rollout in the browser (viser).",
  )
  parser.add_argument(
    "--backend",
    choices=("viser", "native"),
    default="viser",
    help="Viewer to use with --render: browser (viser) or a MuJoCo window.",
  )
  parser.add_argument(
    "--video",
    metavar="PATH",
    help="Save the push rollout as a video (e.g. --video push.mp4).",
  )
  args = parser.parse_args()

  env = BarAngleTrajOptEnv()

  print(f"target angle: {TARGET_ANGLE} rad")
  print(f"hold-spawn-pose cost: {env.evaluate(hold_spawn, TARGET_ANGLE):.4f} rad")

  cost = env.evaluate(
    two_phase_push,
    TARGET_ANGLE,
    render=args.render,
    render_backend=args.backend,
    video_path=args.video,
  )
  print(f"two-phase push cost:  {cost:.4f} rad")
  if args.video:
    print(f"video saved to {args.video}")

  if args.render and args.backend == "viser":
    # The viser server dies with the process; hold it open for inspection.
    input("Viewer is live in the browser -- press Enter to exit. ")


if __name__ == "__main__":
  main()
