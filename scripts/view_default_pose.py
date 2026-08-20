"""View the robot's configured default pose.

``python -m dash_mjlab.robots.dash_constants`` compiles the model but leaves it
at the MJCF's zero pose, which is not the pose the environment spawns in. This
script applies ``EntityCfg.init_state`` (root height plus the joint_pos regex
patterns) and holds it, so what you see is what training actually starts from.

Usage:
  uv run python scripts/view_default_pose.py            # hold the pose, no gravity
  uv run python scripts/view_default_pose.py --physics  # let it fall (no actuators)

On a hybrid-graphics laptop, prefix with:
  __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
"""

import argparse
import re
import time

import mujoco
import mujoco.viewer
import numpy as np
from mjlab.entity.entity import Entity

from dash_mjlab.robots import get_dash_robot_cfg


def resolve_pose(model: mujoco.MjModel, patterns: dict[str, float]) -> dict[str, float]:
  """Resolve joint_pos patterns exactly as mjlab's `resolve_expr` does.

  FIRST match wins, using `re.match` (prefix, not full). Unmatched joints fall
  back to 0.0. Getting this wrong is not academic: resolving later-wins instead
  makes a leading ".*" look harmless when it actually swallows every joint, so
  this script would display a pose the environment never spawns in.
  """
  names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    for i in range(model.njnt)
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
  ]
  pose: dict[str, float] = {}
  for name in names:
    if name is None:
      continue
    pose[name] = 0.0
    for pattern, value in patterns.items():
      if re.match(pattern, name):
        pose[name] = value
        break
  return pose


def add_ground(spec: mujoco.MjSpec) -> None:
  """Add a checkerboard floor at z=0 so spawn clearance is visible.

  The entity spec has no terrain -- mjlab's scene supplies that at env build
  time -- so without this the robot floats in empty space with nothing to judge
  the gap against. Squares are 0.1 m, giving a ruler for the clearance.
  """
  spec.add_texture(
    name="grid_tex",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    width=300,
    height=300,
    rgb1=[0.25, 0.26, 0.28],
    rgb2=[0.32, 0.33, 0.36],
  )
  mat = spec.add_material(name="grid_mat", texrepeat=[20, 20], reflectance=0.1)
  mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid_tex"
  spec.worldbody.add_geom(
    name="ground_plane",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[2.0, 2.0, 0.01],
    pos=[0.0, 0.0, 0.0],
    material="grid_mat",
    contype=1,
    conaffinity=1,
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--physics",
    action="store_true",
    help="Step the simulation. The model has no actuators here, so it collapses; "
    "useful for checking the spawn height, not the pose itself.",
  )
  args = parser.parse_args()

  cfg = get_dash_robot_cfg()
  spec = Entity(cfg).spec
  add_ground(spec)
  model = spec.compile()
  data = mujoco.MjData(model)

  if not args.physics:
    model.opt.gravity[:] = 0.0

  data.qpos[0:3] = cfg.init_state.pos
  data.qpos[3:7] = cfg.init_state.rot
  assert cfg.init_state.joint_pos is not None
  for name, value in resolve_pose(model, cfg.init_state.joint_pos).items():
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[jid]] = value
  mujoco.mj_forward(model, data)

  # Site sits at the sole centre; the box corners are what actually touch down,
  # so measure the lowest corner of each foot collider.
  lowest = float("inf")
  for side in ("l", "r"):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
    pos, mat, size = (
      data.geom_xpos[gid],
      data.geom_xmat[gid].reshape(3, 3),
      (model.geom_size[gid]),
    )
    for sx in (-1, 1):
      for sy in (-1, 1):
        for sz in (-1, 1):
          corner = pos + mat @ (np.array([sx, sy, sz]) * size)
          lowest = min(lowest, float(corner[2]))

  print(f"root height     : {cfg.init_state.pos[2]:.3f} m")
  print(f"lowest foot pt  : {lowest * 1000:+.1f} mm relative to ground")
  if lowest < 0:
    print("  -> SPAWNS INSIDE THE FLOOR. Raise init_state.pos[2].")
  elif lowest > 0.05:
    print(f"  -> spawns falling {lowest * 1000:.0f} mm. Lower init_state.pos[2].")
  else:
    print("  -> OK (0-50 mm clearance)")

  with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
      if args.physics:
        mujoco.mj_step(model, data)
      else:
        # Re-assert the pose so it holds instead of drifting.
        mujoco.mj_forward(model, data)
      viewer.sync()
      time.sleep(model.opt.timestep)


if __name__ == "__main__":
  main()
