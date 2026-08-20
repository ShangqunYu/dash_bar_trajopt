"""Retarget an AMASS (SMPL-H) motion onto Dash, as a CSV for dash-csv-to-npz.

Step 1 of the tracking pipeline:

  1. AMASS .npz  ->  Dash CSV                                  <- this script
  2. dash-csv-to-npz  ->  tracking NPZ
  3. dash-train Mjlab-Tracking-Flat-Dash --env.commands.motion.motion-file ...

Method: rotation retargeting. AMASS stores per-joint local rotations, not 3D
joint positions -- recovering positions would need the SMPL-H body model, which
is a separate registered download. Rotations alone are enough to drive a hinge
robot: each SMPL joint rotation is re-expressed in Dash's axes and decomposed
into the hinge angles of the corresponding Dash chain.

What that buys and costs: no downloads, no extra dependencies, and it follows
the source motion's joint angles closely. It does *not* guarantee foot
placement, because nothing solves for where the feet land -- the feet follow
from the leg angles. Expect some foot skate, and check the contact report this
prints.

Conventions, both established from the data rather than assumed:
  - AMASS world is Z-up (the body's +Y axis lands on world +Z).
  - SMPL canonical body axes are +X left, +Y up, +Z forward; Dash (MuJoCo) uses
    +X forward, +Y left, +Z up. `_SMPL_TO_DASH` is that relabelling.

Usage:
  uv run dash-amass-to-csv \
      --input-file ~/Downloads/CMU/CMU/07/07_01_poses.npz \
      --output-file motion.csv
"""

import dataclasses
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import tyro
from scipy.spatial.transform import Rotation

from dash_mjlab.robots import get_dash_robot_cfg
from dash_mjlab.scripts.csv_to_npz import CSV_JOINT_ORDER

# SMPL-H body joints. The hand joints (22..51) are ignored: Dash has no wrist.
_PELVIS = 0
_L_HIP, _R_HIP = 1, 2
_SPINE1, _SPINE2, _SPINE3 = 3, 6, 9
_L_KNEE, _R_KNEE = 4, 5
_L_ANKLE, _R_ANKLE = 7, 8
_L_COLLAR, _R_COLLAR = 13, 14
_L_SHOULDER, _R_SHOULDER = 16, 17
_L_ELBOW, _R_ELBOW = 18, 19

# Dash's torso spans what SMPL splits across pelvis, three spine joints and the
# collars. Folding that chain into the shoulder is the only way a rigid-torso
# robot can express a human's chest rotation at all.
_ARM_CHAIN = {
  "l": (_SPINE1, _SPINE2, _SPINE3, _L_COLLAR, _L_SHOULDER),
  "r": (_SPINE1, _SPINE2, _SPINE3, _R_COLLAR, _R_SHOULDER),
}
_HIP = {"l": _L_HIP, "r": _R_HIP}
_KNEE = {"l": _L_KNEE, "r": _R_KNEE}
_ANKLE = {"l": _L_ANKLE, "r": _R_ANKLE}
_ELBOW = {"l": _L_ELBOW, "r": _R_ELBOW}

# v_dash = _SMPL_TO_DASH @ v_smpl, i.e. (left, up, fwd) -> (fwd, left, up).
_SMPL_TO_DASH = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@dataclasses.dataclass
class Report:
  """Numbers worth looking at before spending GPU hours on a bad reference."""

  frames: int
  scale: float
  clipped: dict[str, float]
  foot_penetration_mm: float
  foot_clearance_mm: float
  root_height: tuple[float, float]


def _to_dash_axes(rot: Rotation) -> Rotation:
  """Re-express a rotation given in SMPL body axes in Dash's body axes."""
  return Rotation.from_matrix(_SMPL_TO_DASH @ rot.as_matrix() @ _SMPL_TO_DASH.T)


def _compose(rots: list[Rotation]) -> Rotation:
  out = rots[0]
  for rot in rots[1:]:
    out = out * rot
  return out


# Euler order per chain, in the order the hinges physically appear on Dash, and
# the joint each term feeds. Single-hinge chains keep only the leading term.
_CHAIN_EULER: dict[str, tuple[str, tuple[str, ...]]] = {
  "hip": ("ZXY", ("hip_yaw", "hip_roll", "hip_pitch")),
  "knee": ("YXZ", ("knee_pitch",)),
  "ankle": ("YXZ", ("ankle_pitch",)),
  "arm": ("YXZ", ("shoulder_pitch", "shoulder_roll", "shoulder_yaw")),
  "elbow": ("YXZ", ("elbow_pitch",)),
}


def _chain_rotations(poses: np.ndarray) -> dict[str, Rotation]:
  """The SMPL rotation driving each Dash chain, expressed in Dash's axes."""
  local = [
    Rotation.from_rotvec(poses[:, 3 * j : 3 * j + 3])
    for j in range(poses.shape[1] // 3)
  ]
  out: dict[str, Rotation] = {}
  for side in ("l", "r"):
    out[f"{side}_hip"] = _to_dash_axes(local[_HIP[side]])
    out[f"{side}_knee"] = _to_dash_axes(local[_KNEE[side]])
    out[f"{side}_ankle"] = _to_dash_axes(local[_ANKLE[side]])
    out[f"{side}_arm"] = _to_dash_axes(_compose([local[j] for j in _ARM_CHAIN[side]]))
    out[f"{side}_elbow"] = _to_dash_axes(local[_ELBOW[side]])
  return out


def retarget_angles(poses: np.ndarray, center: bool = True) -> dict[str, np.ndarray]:
  """Map SMPL local joint rotations onto Dash's hinge angles.

  Each Dash chain is read as an intrinsic Euler decomposition in the order the
  hinges physically appear: hips are yaw(Z) then roll(X) then pitch(Y), arms are
  pitch(Y) then roll(X) then yaw(Z).

  With ``center``, each chain is first referred to its own mean rotation over
  the clip, so what comes out is the motion *about the human's neutral pose* for
  that take. This is done in SO(3), before the Euler step, and that ordering is
  not cosmetic: SMPL's rest pose is a T-pose, so arms-at-the-side is already
  ~90 degrees of shoulder roll, which sits exactly on the Y-X-Z decomposition's
  singularity. Decomposing first and subtracting a median afterwards puts the
  shoulder terms right in gimbal lock, where they flip by 2*pi between frames --
  measured on CMU walking clips, that turned a +-0.4 rad arm swing into a
  spurious +-4 rad range. Referring the rotation first keeps the decomposition
  near identity, where it is well conditioned.

  Approximation worth knowing about: Dash's leg links carry fixed rotations in
  the MJCF (the hip is tilted ~25 degrees about X, the thigh ~19 about Y), so
  its hinge axes are not exactly the torso frame's Z/X/Y. The decomposition
  ignores that tilt. It is a rest-pose offset, so relative motion still maps
  sensibly, but this is the first thing to revisit if a limb looks skewed.
  """
  out: dict[str, np.ndarray] = {}
  for chain, rot in _chain_rotations(poses).items():
    side, kind = chain.split("_", 1)
    if center:
      rot = rot.mean().inv() * rot
    order, joints = _CHAIN_EULER[kind]
    euler = rot.as_euler(order)
    for i, joint in enumerate(joints):
      out[f"{side}_{joint}"] = euler[:, i]
  return out


def _nominal_pose(model: mujoco.MjModel) -> np.ndarray:
  """Dash's configured stance, in CSV_JOINT_ORDER."""
  import re

  patterns = get_dash_robot_cfg().init_state.joint_pos
  assert patterns is not None
  nominal = np.zeros(len(CSV_JOINT_ORDER))
  for i, name in enumerate(CSV_JOINT_ORDER):
    for pattern, value in patterns.items():
      if re.match(pattern, name):
        nominal[i] = value
        break
  return nominal


def _lowest_foot_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
  """Lowest corner of either foot collider, in world z."""
  lowest = np.inf
  for side in ("l", "r"):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
    pos, mat, size = (
      data.geom_xpos[gid],
      data.geom_xmat[gid].reshape(3, 3),
      model.geom_size[gid],
    )
    for sx in (-1, 1):
      for sy in (-1, 1):
        for sz in (-1, 1):
          corner = pos + mat @ (np.array([sx, sy, sz]) * size)
          lowest = min(lowest, float(corner[2]))
  return lowest


def build_csv(
  amass_file: str,
  output_fps: float,
  gain: float,
  pose_offset: bool,
  scale: float | None,
  ground: str = "per_frame",
) -> tuple[np.ndarray, Report]:
  raw = np.load(amass_file)
  poses, trans = raw["poses"], raw["trans"]
  source_fps = float(raw["mocap_framerate"])

  step = max(1, int(round(source_fps / output_fps)))
  poses, trans = poses[::step], trans[::step]
  num_frames = len(poses)

  angles = retarget_angles(poses, center=pose_offset)

  model = get_dash_robot_cfg()
  from mjlab.entity.entity import Entity

  mj_model = Entity(model).spec.compile()
  mj_data = mujoco.MjData(mj_model)
  nominal = _nominal_pose(mj_model)

  # Anchor the human motion on Dash's stance instead of mapping angles
  # absolutely, for two reasons that both bite hard:
  #
  #  - SMPL's rest pose is a T-pose, so a human standing with arms down is
  #    already ~75 degrees of shoulder roll away from zero. Mapped absolutely,
  #    that offset alone pins Dash's arms (roll range is only +-0.3) for 100%
  #    of frames before the motion contributes anything.
  #  - Dash's stance is a crouch (knee 0.70), because its ankle cannot reach
  #    neutral, whereas a human's neutral is straight-legged.
  #
  # The per-clip median is the human's neutral for that take, so subtracting it
  # lines the two neutrals up and leaves the motion as a delta on top. It also
  # discards any constant bias in the take -- a permanent forward lean is lost,
  # which for locomotion is a good trade.
  columns = []
  for i, name in enumerate(CSV_JOINT_ORDER):
    value = gain * angles[name]
    columns.append(nominal[i] + value if pose_offset else value)
  joint_angles = np.stack(columns, axis=1)

  # Clip to the model's own limits and record how much was lost; a reference
  # that spends most of its time on a limit is not one worth training against.
  jids = [
    mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in CSV_JOINT_ORDER
  ]
  limits = np.array([mj_model.jnt_range[j] for j in jids])
  clipped_frac = {
    name: float(
      np.mean(
        (joint_angles[:, i] < limits[i, 0] - 1e-6)
        | (joint_angles[:, i] > limits[i, 1] + 1e-6)
      )
    )
    for i, name in enumerate(CSV_JOINT_ORDER)
  }
  joint_angles = np.clip(joint_angles, limits[:, 0], limits[:, 1])

  # Root orientation: world_from_dash = world_from_smpl @ relabelling.
  root = Rotation.from_rotvec(poses[:, :3])
  root_dash = Rotation.from_matrix(root.as_matrix() @ _SMPL_TO_DASH.T)
  quat_xyzw = root_dash.as_quat()

  # Root translation: a 0.69 m robot cannot take a 1.7 m human's stride, so
  # shrink the trajectory by the hip-height ratio. Height itself is fixed by the
  # ground pass below, so only the horizontal scale really matters here.
  hip_height = float(get_dash_robot_cfg().init_state.pos[2])
  if scale is None:
    scale = hip_height / float(np.median(trans[:, 2]))
  root_pos = trans * scale
  root_pos[:, 2] = trans[:, 2] * scale

  # Ground pass. Per frame, not once for the clip: the human's pelvis height
  # scaled by a single number does not put Dash's legs -- different proportions,
  # different link lengths -- on the floor, and offsetting by the clip's single
  # lowest frame leaves the robot hovering for the rest of it. Measured across
  # CMU walking clips, a clip-wide offset had a foot within 10 mm of the ground
  # in only 3-32% of frames.
  #
  # Re-anchoring each frame on its own lowest foot does not flatten the gait:
  # the root's height *above the stance foot* is untouched, so the bounce
  # survives; only the arbitrary global offset goes. It does remove flight
  # phases, since some foot is always grounded -- use "clip" for running.
  qpos_joint_ids = [mj_model.jnt_qposadr[j] for j in jids]
  lowest = np.empty(num_frames)
  for f in range(num_frames):
    mj_data.qpos[0:3] = root_pos[f]
    mj_data.qpos[3:7] = np.roll(quat_xyzw[f], 1)  # xyzw -> wxyz
    for k, adr in enumerate(qpos_joint_ids):
      mj_data.qpos[adr] = joint_angles[f, k]
    mujoco.mj_forward(mj_model, mj_data)
    lowest[f] = _lowest_foot_z(mj_model, mj_data)
  if ground == "per_frame":
    root_pos[:, 2] -= lowest
    lowest = np.zeros_like(lowest)
  else:
    root_pos[:, 2] -= lowest.min()
    lowest -= lowest.min()

  csv = np.concatenate([root_pos, quat_xyzw, joint_angles], axis=1)
  report = Report(
    frames=num_frames,
    scale=scale,
    clipped=clipped_frac,
    foot_penetration_mm=float(max(0.0, -lowest.min()) * 1000),
    foot_clearance_mm=float(lowest.mean() * 1000),
    root_height=(float(root_pos[:, 2].min()), float(root_pos[:, 2].max())),
  )
  return csv, report


def main(
  input_file: str,
  output_file: str,
  output_fps: int = 30,
  gain: float = 1.0,
  pose_offset: bool = True,
  scale: float | None = None,
  ground: Literal["per_frame", "clip"] = "per_frame",
) -> None:
  """Retarget one AMASS clip onto Dash.

  Args:
    input_file: An AMASS SMPL-H `*_poses.npz`.
    output_file: CSV to write, in dash-csv-to-npz's column order.
    output_fps: Rate to decimate the 120 Hz source to. Pass the same number to
      dash-csv-to-npz as --input-fps.
    gain: Scales the human joint excursions. Below 1.0 tames motions that
      overrun Dash's limits.
    pose_offset: Treat the human angles as a delta on Dash's nominal stance
      rather than absolute angles. See build_csv for why this is the default.
    scale: Root translation scale. Defaults to Dash's hip height over the
      clip's median pelvis height.
    ground: How the root height is anchored. "per_frame" keeps the lowest foot
      on the floor every frame; "clip" offsets once by the clip minimum, which
      preserves flight phases but leaves the robot hovering during walking.
  """
  csv, report = build_csv(input_file, output_fps, gain, pose_offset, scale, ground)

  out = Path(output_file)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(out, csv, delimiter=",", fmt="%.6f")

  print(f"\nWrote {out}")
  print(f"  frames        : {report.frames} @ {output_fps} fps")
  print(f"  root scale    : {report.scale:.3f}")
  print(
    f"  root height   : {report.root_height[0]:.3f} - {report.root_height[1]:.3f} m"
  )
  print(f"  foot clearance: {report.foot_clearance_mm:.1f} mm mean above ground")

  worst = sorted(report.clipped.items(), key=lambda kv: -kv[1])
  hit = [(n, f) for n, f in worst if f > 0.01]
  if hit:
    print("\n  clipped against joint limits (fraction of frames):")
    for name, frac in hit:
      flag = "  <-- reference is mostly pinned" if frac > 0.5 else ""
      print(f"    {name:20s} {frac:6.1%}{flag}")
    print("  Lower --gain to reduce, at the cost of a less faithful motion.")
  else:
    print("\n  no joint spends >1% of frames on a limit.")

  print("\nNext:")
  print(f"  uv run dash-csv-to-npz --input-file {out} \\")
  print(f"      --output-file {out.with_suffix('.npz')} --input-fps {output_fps:g}")


def entry() -> None:
  tyro.cli(main)


if __name__ == "__main__":
  entry()
