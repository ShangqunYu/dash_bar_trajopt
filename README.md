# dash_mjlab

For me i need to add __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia when not using viser.

## IL

**Clip viz:** __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
  uv run dash-view-motion --input-file 12_01.csv --speed 1

**Train:** uv run dash-train Mjlab-Tracking-Flat-Dash --env.commands.motion.motion-file motions/12_01.npz --env.scene.num-envs 4096 --agent.logger tensorboard

**Policy viz:** __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia uv run dash-play Mjlab-Tracking-Flat-Dash --checkpoint-file logs/rsl_rl/dash_tracking/timestamp/model_N.pt --motion-file motions/12_01.npz --num-envs 1


## RL

**Train:** uv run dash-train Mjlab-Velocity-Flat-Dash --env.scene.num-envs 4096 --agent.logger tensorboard --agent.max-iterations 10000

**Tensorboard:** uv run tensorboard --logdir logs/rsl_rl/dash_velocity/timestamp

**Play:** uv run dash-play Mjlab-Velocity-Flat-Dash \
  --checkpoint-file logs/rsl_rl/dash_velocity/timestamp/model_N.pt \
  --num-envs 4 --viewer viser


Dash humanoid assets and RL tasks built on
[mjlab](https://github.com/mujocolab/mjlab).

Dash is an 18-DOF humanoid: 5 per leg (hip yaw/roll/pitch, knee pitch, ankle pitch)
and 4 per arm (shoulder pitch/roll/yaw, elbow pitch), ~34 kg.

There is also an upper-body-only variant -- torso and both arms, 8 DOF, no legs
and no hip joints, welded to the world -- used for the tabletop manipulation
task. See [Upper body](#upper-body) below.

## Install

```sh
uv sync --extra cu128 # or --extra cpu on a machine w/o an NVIDIA GPU
```

## Usage

```sh
uv run dash-list-envs

uv run dash-train Mjlab-Velocity-Flat-Dash
uv run dash-play Mjlab-Velocity-Flat-Dash --wandb-run-path <run>
```

mjlab's own `train` / `play` only import `mjlab.tasks`, so they will not see
Dash. The `dash-` commands register the Dash tasks first, then hand off.

Registered tasks:

| Task | Notes |
|---|---|
| `Mjlab-Velocity-Flat-Dash` | Velocity tracking on a plane. Start here. |
| `Mjlab-Velocity-Rough-Dash` | Velocity tracking on generated rough terrain. |
| `Mjlab-Tracking-Flat-Dash` | Motion imitation. **Needs reference motion data**. |
| `Mjlab-Tracking-Flat-Dash-No-State-Estimation` | Same, w/o linear velocity / anchor observations. |
| `Mjlab-Lift-Box-Dash-UpperBody` | Bimanual pick-and-place on a table. Upper body only. |

To inspect the robot on its own:

```sh
uv run python -m dash_mjlab.robots.dash_constants             # MuJoCo viewer
uv run python -m dash_mjlab.robots.dash_upper_body_constants  # upper body only
uv run dash-export-scene Mjlab-Velocity-Flat-Dash --output-dir /tmp/dash
```

## Upper body

`Mjlab-Lift-Box-Dash-UpperBody` trains a policy to pick a box off one corner of
a table, carry it across, and set it down on the opposite corner, using both
arms.

```sh
uv run dash-train Mjlab-Lift-Box-Dash-UpperBody --env.scene.num-envs 4096 \
  --agent.logger tensorboard
uv run dash-play Mjlab-Lift-Box-Dash-UpperBody \
  --checkpoint-file logs/rsl_rl/dash_lift_box/timestamp/model_N.pt --num-envs 4
```

The robot is `xmls/dash_upper_body.xml`: the torso and both arms lifted out of
`dash.xml` unchanged, with the leg chains deleted rather than actuator-disabled,
so they cost nothing in qpos, contacts or observations. Torso carries no
freejoint, which makes it a fixed-base entity; mjlab wraps those in a mocap body
automatically so it is still placed per-environment.

4 things about this task are not obvious, and all are load-bearing:

Because the forearms are the grip surface, they carry the grip friction
(`HAND_COLLISION`). Note this **inverts** `dash_constants`, where the same
capsules are frictionless so that grazing self-contacts don't fight the gait.
Don't copy these values back to the walking robot.

**The box is 20 cm wide, and widening it is what buys workspace.** Simulating
the squeeze across a grid gives a usable strip of x in [0.24, 0.32] with a 16 cm
box, against x in [0.22, 0.32] at 20 cm. Widening moves the *near* limit in,
because holding the hands further apart is the easy direction for these
shoulders; the far limit is the arm simply running out of forward reach and does
not move. The wider box also tilts less in transit, since the two contact points
sit further apart and resist rotation better.

**The workspace is a 10 x 12 cm patch, and what bounds the near edge is torque,
not reach.** Outside the strip the grip does not weaken, it fails outright -- at
x = 0.34 the hands never touch the box. IK solves the nearer poses exactly; the
problem is that shoulder roll is already pinned at its 0.3 rad limit there and
the servos still settle 6-11 cm wider than commanded under gravity alone, with
no box in the way. Reaching in toward the chest is the weakest thing these arms
do.

**"Across the table" is 12 cm, and the table is sized to say so.** The torso is
welded and there is no waist yaw, so the whole two-hand workspace sits straight
in front of the chest; a left-to-right traverse of a normal desk does not exist
for this robot. The traverse runs diagonally across the strip instead, and the
table is 30 x 46 cm so a corner-to-corner carry genuinely spans it. A custom
command term (`TraverseLiftingCommand`) samples the pick and the place at
opposite corners as a matched pair, randomizing both the end and the side, so
every episode travels 10-14 cm rather than the ~7 cm mean you get from sampling
the two independently. Both ends are on the table surface: this is a place, not
a hold-in-the-air.

The box is repositioned only at the *start* of an episode. mjlab's stock
`LiftingCommand` also re-spawns the object every time its resample timer fires,
which is harmless for a 20 g cube and a gripper but not here: once the policy
can actually grasp, the timer drops the box inside two squeezing hands, which
measures as a 217 N interpenetration spike against a normal grip of 30-75 N, and
throws away the credit for the grasp in progress. A mid-episode resample here
moves only the goal, turning it into "now take it back the other way".

`tests/test_dash_upper_body_entity.py::test_grip_forms_across_the_traverse`
re-derives all of this by simulating a squeeze at every corner of the pick and
place regions, so an MJCF edit that invalidates them fails loudly. It simulates
rather than checking reach on purpose: an earlier position-only version of that
test passed a pick zone where no grip was possible at all.

## Layout

```
src/dash_mjlab/
  robots/
    dash_constants.py             EntityCfg: actuators, collisions, spawn pose.
    dash_upper_body_constants.py  Same, for the 8-DOF upper body.
    xmls/dash.xml                 MJCF.
    xmls/dash_upper_body.xml      Torso + arms, fixed base, no hands.
    xmls/assets/                  Decimated STL meshes.
  tasks/
    velocity/            Velocity tracking (flat + rough).
    tracking/            Motion imitation.
    manipulation/        Bimanual pick-and-place (upper body).
  scripts/               CLI entry points.
scripts/decimate_meshes.py   Regenerates xmls/assets from raw CAD STLs.
```

## Sim2real gaps

**Effort limits are estimates.** `dash_constants.py` scales them from the
robot's mass because neither the URDF nor the DashMotorControl firmware records
real per-joint torque limits (`KT` and gear ratio are runtime registers there).
Replace `EFFORT_LIMIT_*` once real numbers exist.

**Collision primitives are approximations.** The foot boxes come from the 20 dof xml. 
Everything else (torso box, limb capsules) was fitted to the visual mesh bounds and 
is a first pass. Note the foot box bottom sits ~7.5 mm below the foot mesh, so the 
visual appears to float slightly; that offset is upstream's, left as-is.

**Change IMU location.** `imu_in_torso` should be moved to the real mounting spot

**The gripping forearms are not hardware.** `Mjlab-Lift-Box-Dash-UpperBody`
gives the forearm capsules a friction they do not have on the real robot (1.2
slide nominal, randomized 0.8-1.5, plausible for rubber on cardboard but not
measured). A policy trained on it transfers only if the forearm ends are
actually rubber-coated. The upper-body arm stiffnesses are also raised well
above the walking config's, which is right for a fixed torso carrying a load and
wrong for a walking robot -- don't carry them back. The upper-body arm stiffnesses are
also raised well above the walking config's, which is right for a fixed torso
carrying a load and wrong for a walking robot -- don't carry them back.

## Checks

```sh
uv run ruff format && uv run ruff check --fix
uv run pyright
uv run pytest tests/
```
