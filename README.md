# dash_bar_trajopt

Trajectory-optimization benchmark on the Dash humanoid's upper body: a
horizontal bar sits on a vertical revolute joint in front of the robot, so it
can only swing in the horizontal plane and gravity never moves it. The goal is
to make the bar point at a commanded angle by pushing it with the right arm.

You provide four joint-position trajectories (plain Python functions of time),
the simulator rolls them out with PD control, and you get back a scalar cost.
Deterministic: the same trajectories always produce the same cost.

## Setup

Install [uv](https://docs.astral.sh/uv/), then from the repo root:

```bash
uv sync
```

## Usage

```python
from dash_mjlab.trajopt import BarAngleTrajOptEnv

env = BarAngleTrajOptEnv()          # horizon=5.0 s, kp=20, kd=1, dt=0.005 s

cost = env.evaluate(
  [f_shoulder_pitch, f_shoulder_roll, f_shoulder_yaw, f_elbow_pitch],
  target_angle=0.6,                 # rad
)
```

- Each `f(t)` maps time in seconds (`0 <= t < horizon`) to a desired joint
  position in radians. Values outside the joint's range are clamped;
  non-finite values raise.
- The four functions drive the right arm, in this order:
  `r_shoulder_pitch`, `r_shoulder_roll`, `r_shoulder_yaw`, `r_elbow_pitch`
  (ranges: `[-0.6, 1.1]`, `[-0.6, 0.3]`, `[-0.8, 0.8]`, `[-1.5, 0]`).
  The left arm is held at its spawn pose and never reaches the bar.
- Tracking is a PD torque law, `kp * (desired - q) - kd * qd`, clamped to the
  30 Nm effort limit. `kp`, `kd`, `horizon`, `timestep` and the bar's initial
  angle are constructor arguments.
- **Cost** = `|shortest angular distance(target_angle, final bar angle)|` in
  radians, measured once when the horizon runs out. Doing nothing costs
  `|target_angle|`; the bar starts at angle 0 (pointing at the robot),
  positive angles swing its free end to the robot's right.

A 5 s rollout evaluates in ~30 ms on CPU, so search methods can afford
thousands of candidates. One `BarAngleTrajOptEnv` instance is reusable across
evaluations but not thread-safe; create one per worker for parallel search.

### Debugging a candidate

```python
cost, traj = env.evaluate(fns, target_angle=0.6, return_trajectory=True)
# traj["time"], traj["bar_angle"], traj["joint_pos"], traj["joint_target"]

cost = env.evaluate(fns, target_angle=0.6, render=True)   # watch it live
```

Rendering is browser-based (viser): the first rendered call starts a local
server and prints its URL (default `http://localhost:8080`); open it in a
browser and later rendered calls stream into the same page. This also works
over SSH with port forwarding. Pass `render_backend="native"` for a classic
MuJoCo window instead. In either viewer, the translucent green bar shows the
commanded target angle.

## Example

```bash
uv run python -m dash_mjlab.trajopt.example            # print costs
uv run python -m dash_mjlab.trajopt.example --render   # watch it in the browser
```

The example rolls out a two-phase push (reach in past the bar, then sweep
outward carrying it) that lands the bar within 0.005 rad of a -0.5 rad
target, against 0.5 for doing nothing -- it was found by plain random search
over a smoothstep parameterization, so it is also a floor for what your
optimizer should beat.

## Tests

```bash
uv run pytest tests/test_bar_trajopt.py
```
