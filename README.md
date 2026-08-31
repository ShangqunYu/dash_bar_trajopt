# dash_bar_trajopt

Trajectory-optimization benchmark on the Dash humanoid's upper body: a flat
ship's wheel -- three horizontal spokes on one vertical revolute joint -- sits
in front of the robot, so it can only spin in the horizontal plane and gravity
never moves it. The goal is to make the red spoke point at a commanded angle
by pushing the wheel with the right arm.

You provide one vectorized control law (phase in, normalized joint targets
out), the simulator rolls it out with PD control, and you get back a scalar
cost. Deterministic: the same law always produces the same cost.

## Setup

Install [uv](https://docs.astral.sh/uv/), then from the repo root:

```bash
uv sync
```

## Usage

```python
from dash_mjlab.trajopt import BarAngleTrajOptEnv

env = BarAngleTrajOptEnv()          # horizon=5.0 s, kp=20, kd=1, dt=0.005 s

cost = env.evaluate(control_law, target_angle=0.6)   # rad
```

- `control_law` maps phase in `[0, 1)` (the fraction of the horizon elapsed,
  shape `(..., 1)`) to normalized joint targets in `[-1, 1]` (shape
  `(..., 4)`): 0 holds the spawn pose, +-1 reaches the joint's upper/lower
  limit, so the zero function is the do-nothing law and no radians or joint
  limits ever reach the optimizer. Values outside the cube are clipped;
  non-finite values raise. It must be vectorized over the leading axes --
  returning the wrong shape is an error, not a silently broadcast constant.
- By default the law is queried once for the whole horizon before stepping.
  Pass `precompute=False` to have it queried one step at a time instead; for a
  law that is a function of time alone the two are equivalent, and the costs
  agree exactly.
- The four output columns drive the right arm, in this order:
  `r_shoulder_pitch`, `r_shoulder_roll`, `r_shoulder_yaw`, `r_elbow_pitch`.
  The left arm is held at its spawn pose and never reaches the wheel.
- Tracking is a PD torque law, `kp * (desired - q) - kd * qd`, clamped to the
  30 Nm effort limit. `kp`, `kd`, `horizon`, `timestep`, the wheel's initial
  angle, its spoke count and the reaching-penalty weight are constructor
  arguments (`num_spokes=1` recovers a single lone bar).
- **Cost** = `|shortest angular distance(target_angle, final wheel angle)|` in
  radians, measured on the red spoke when the horizon runs out. The wheel
  starts at angle 0 (red spoke pointing at the robot); positive angles swing
  its free end to the robot's right.
- **Dense reaching term**: a rollout whose end-effector never touches the
  wheel is additionally charged `reach_penalty_weight` (default 1 rad/m) times
  its closest approach to the wheel, so never-touching candidates are ordered
  by how close they came instead of all costing exactly `|target_angle|`.
  Contact at any point during the rollout removes the term entirely -- it
  never trades off against the angle error.

A 5 s rollout evaluates in ~30 ms on CPU, so search methods can afford
thousands of candidates. One `BarAngleTrajOptEnv` instance is reusable across
evaluations but not thread-safe; create one per worker for parallel search.

### Debugging a candidate

```python
cost, traj = env.evaluate(control_law, target_angle=0.6, return_trajectory=True)
# traj["time"], traj["bar_angle"], traj["joint_pos"], traj["joint_target"],
# traj["hand_to_wheel"], traj["touched"]

cost = env.evaluate(control_law, target_angle=0.6, render=True)   # watch it live

cost = env.evaluate(control_law, target_angle=0.6, video_path="push.mp4")
```

Video capture is offscreen (no window, works headless and at full speed);
`video_fps` and `video_size` are also accepted.

Rendering is browser-based (viser): the first rendered call starts a local
server and prints its URL (default `http://localhost:8080`). The rollout
waits until a browser is connected before it starts playing; later rendered
calls stream into the same page. This also works
over SSH with port forwarding. Pass `render_backend="native"` for a classic
MuJoCo window instead. In either viewer, the translucent green bar shows the
commanded target angle.

## Example

```bash
uv run python -m dash_mjlab.trajopt.example            # print costs
uv run python -m dash_mjlab.trajopt.example --render           # watch in browser
uv run python -m dash_mjlab.trajopt.example --video push.mp4   # save a video
```

The example rolls out a two-phase push (reach in past the bar, then sweep
outward carrying it) that lands the bar within 0.005 rad of a -0.5 rad
target, against 0.5 for doing nothing -- it was found by plain random search
over a smoothstep parameterization, so it is also a floor for what your
optimizer should beat.

## Progress dashboard

```bash
uv run python scripts/run_fbo.py --num-spokes 3   # logs to outputs/turn-3-spoke-to-61p352/
uv run python scripts/make_fbo_dashboard.py outputs/turn-3-spoke-to-61p352
```

The standard grid is 1/3/6 spokes crossed with an easy and a hard target
(hold and one-push demo costs from the run log headers):

| `--num-spokes` | `--target-angle` | hold cost | one-push demo |
| --- | --- | --- | --- |
| 1 | `-0.5` (-29 deg) | 0.70 | 0.12 |
| 3 | `-0.5` (-29 deg) | 0.64 | 0.0003 |
| 6 | `-0.5` (-29 deg) | 0.63 | 0.19 |
| 1 | `3.0` (172 deg) | 3.20 | 2.90 |
| 3 | `3.0` (172 deg) | 3.14 | 2.78 |
| 6 | `3.0` (172 deg) | 3.13 | 2.59 |

`-0.5` is within a single push of the spawn pose. `3.0` is past any wheel's
spoke spacing: the one-push demo runs its spoke out of the arm's sweep and
stalls, so reaching the target needs the arm to release and pick up the next
spoke -- and the more spokes, the sooner a fresh one swings into reach.

`--num-spokes` takes any count from 1 up (1 is the original lone bar); the
spawn pose clears the wheel by at least 11 cm through 12 spokes, and the
runner warns if a count ever puts a spoke against a parked hand. The count is
recorded in the log header, so `render_eval.py` rebuilds the right wheel.

```bash
mkdir -p .slurm-logs
sbatch slurm/run_parallel_sweep.sbatch  # all six grid cells on one node
```

The runner logs every rollout (cost, wall time, law curve) to
`<out>/log.jsonl` as it goes (with no `--out` it names the directory
`outputs/turn-<spokes>-spoke-to-<deg>`, the target as a bar heading in
degrees) and renders a video per running best
when done; the builder bakes the log and videos into a self-contained HTML
page from `scripts/fbo_dashboard_template.html`. It works on a partial log
too, so you can build the page mid-run.

## Tests

```bash
uv run pytest tests/test_bar_trajopt.py
```
