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
exists to serve -- random search over the two-phase smoothstep family in
``trajopt.search``, refined around the best sample. Its two legs overlap into
one continuous motion that hooks the wheel and drags spoke 0 onto the target,
scoring 0.001 rad against a target of -0.5. Doing nothing scores 0.5 plus the
reaching penalty (the closest-approach distance to the wheel, ~0.11 m), since
a rollout that never touches the wheel is charged for how far it stayed from
it.
"""

import argparse

import numpy as np
from jaxtyping import Float

from dash_mjlab.trajopt import BarAngleTrajOptEnv

TARGET_ANGLE = -0.5
# Searched on the 3-spoke wheel (see scratch search over TwoPhaseSmoothstep):
# the legs overlap -- leg B's quick yaw-in starts first, leg A's slower reach
# rides on top -- and together they hook a spoke and drag the wheel onto the
# target.
POSE_A = np.array([-0.1423324, 0.3257772, -0.7219670, -0.8735937])
POSE_B = np.array([-0.5906081, 0.6926238, 0.9551563, -0.1813463])
S_A, W_A = 0.5468864, 0.7497688  # leg A start and duration, phase fractions
S_B, W_B = 0.3752457, 0.0191130  # leg B start and duration


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
