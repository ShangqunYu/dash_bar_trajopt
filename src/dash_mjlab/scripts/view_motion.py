"""Play back a retargeted motion on Dash, before training anything on it.

Takes either stage of the pipeline:

  .csv  -- the retargeter's output, base pose + 18 joint angles per frame
  .npz  -- dash-csv-to-npz's output, replayed from its logged root and joints

This drives the model's qpos straight from the file, so what you see is the
reference itself, not a policy's attempt at it. `dash-play --agent zero` is the
other option, but that shows the tracking command's ghost frames next to a robot
that immediately falls over; to judge whether a retarget is any good you want
the motion on the robot.

Watch for the two things rotation retargeting gets wrong: feet sliding while
they should be planted, and soles that never go flat.

Usage:
  uv run dash-view-motion --input-file motion.csv
  uv run dash-view-motion --input-file motion.npz --speed 0.5

On a hybrid-graphics laptop, prefix with:
  __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
"""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import tyro
from mjlab.entity.entity import Entity

from dash_mjlab.robots import get_dash_robot_cfg
from dash_mjlab.scripts.csv_to_npz import CSV_JOINT_ORDER


def add_ground(spec: mujoco.MjSpec) -> None:
  """A 0.1 m checkerboard, so foot contact and stride length are readable."""
  spec.add_texture(
    name="grid_tex",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    width=300,
    height=300,
    rgb1=[0.25, 0.26, 0.28],
    rgb2=[0.32, 0.33, 0.36],
  )
  mat = spec.add_material(name="grid_mat", texrepeat=[80, 80], reflectance=0.1)
  mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid_tex"
  spec.worldbody.add_geom(
    name="ground_plane",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[8.0, 8.0, 0.01],
    pos=[0.0, 0.0, 0.0],
    material="grid_mat",
    contype=0,
    conaffinity=0,
  )


def load_frames(
  path: Path, model: mujoco.MjModel
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
  """Return (root_pos, root_quat_wxyz, joint_angles, fps) for either format."""
  if path.suffix == ".npz":
    data = np.load(path, allow_pickle=True)
    # body 0 is the root; dash-csv-to-npz logs every body's world pose, and the
    # joint columns are already in the model's own order.
    return (
      data["body_pos_w"][:, 0],
      data["body_quat_w"][:, 0],
      data["joint_pos"],
      float(np.asarray(data["fps"]).ravel()[0]),
    )

  csv = np.loadtxt(path, delimiter=",")
  if csv.shape[1] != 7 + len(CSV_JOINT_ORDER):
    raise SystemExit(
      f"{path} has {csv.shape[1]} columns, expected {7 + len(CSV_JOINT_ORDER)} "
      "(3 base pos, 4 base quat xyzw, 18 joint angles)."
    )
  # CSV joints follow CSV_JOINT_ORDER; qpos follows the model. Reorder rather
  # than assume they agree.
  order = [
    model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    for n in CSV_JOINT_ORDER
  ]
  angles = np.zeros((len(csv), model.nq - 7))
  for column, adr in enumerate(order):
    angles[:, adr - 7] = csv[:, 7 + column]
  return csv[:, 0:3], np.roll(csv[:, 3:7], 1, axis=1), angles, 30.0


def main(input_file: str, speed: float = 1.0, fps: float | None = None) -> None:
  """Replay a retargeted motion in the MuJoCo viewer, looping.

  Args:
    input_file: A .csv from dash-amass-to-csv or a .npz from dash-csv-to-npz.
    speed: Playback rate multiplier. Below 1.0 to study foot contact.
    fps: Override the frame rate. CSVs carry none, so they assume 30.
  """
  path = Path(input_file).expanduser()
  spec = Entity(get_dash_robot_cfg()).spec
  add_ground(spec)
  model = spec.compile()
  data = mujoco.MjData(model)

  root_pos, root_quat, angles, file_fps = load_frames(path, model)
  rate = fps if fps is not None else file_fps
  num_frames = len(root_pos)

  print(
    f"\n{path.name}: {num_frames} frames @ {rate:g} fps "
    f"({num_frames / rate:.1f} s), playing at {speed:g}x"
  )
  print(f"  root height : {root_pos[:, 2].min():.3f} - {root_pos[:, 2].max():.3f} m")
  print(f"  travel      : {np.linalg.norm(root_pos[-1, :2] - root_pos[0, :2]):.2f} m")
  print("\nLooping. Close the window to stop.")

  frame = 0
  with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
      start = time.time()
      data.qpos[0:3] = root_pos[frame]
      data.qpos[3:7] = root_quat[frame]
      data.qpos[7:] = angles[frame]
      # Kinematics only: the reference is a pose sequence, and stepping physics
      # would just make the robot fall away from it.
      mujoco.mj_forward(model, data)
      viewer.sync()

      frame = (frame + 1) % num_frames
      remaining = (1.0 / rate) / max(speed, 1e-6) - (time.time() - start)
      if remaining > 0:
        time.sleep(remaining)


def entry() -> None:
  tyro.cli(main)


if __name__ == "__main__":
  entry()
