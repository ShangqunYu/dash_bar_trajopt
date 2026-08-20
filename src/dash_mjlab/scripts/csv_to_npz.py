"""Convert a retargeted Dash motion CSV into the NPZ the tracking task loads.

Step 2 of the BeyondMimic pipeline mjlab implements:

  1. retarget source mocap (e.g. AMASS) onto Dash  ->  CSV
  2. replay that CSV in MuJoCo Warp, recording every body's pose and velocity
     -> NPZ                                                    <- this script
  3. train:  dash-train Mjlab-Tracking-Flat-Dash \
               --env.commands.motion.motion-file <npz>

mjlab ships its own ``mjlab.scripts.csv_to_npz``, but it is wired to the G1: it
builds the G1 scene and expects that robot's 29 joints as CSV columns. The NPZ
stores body arrays indexed by *body number*, so a file built against the wrong
model silently maps tracking targets onto the wrong links. Hence a Dash copy.

The heavy lifting (CSV parsing, resampling, finite-difference velocities) is
imported from mjlab so the two stay consistent; only the replay and the save
are reimplemented, to write a plain local file instead of uploading to WandB.

Input CSV, one row per frame, no header:

  base_x, base_y, base_z, qx, qy, qz, qw, <18 joint angles>

Base quaternion is **xyzw** (Unitree's convention, which mjlab's loader flips to
wxyz internally). Joint angles are in radians, in CSV_JOINT_ORDER below.

Usage:
  uv run dash-csv-to-npz --input-file motion.csv --output-file motion.npz \
      --input-fps 30 --output-fps 50
"""

from pathlib import Path
from typing import Any

import mjlab
import numpy as np
import torch
import tyro
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader
from mjlab.sim.sim import Simulation, SimulationCfg
from tqdm import tqdm

from dash_mjlab.tasks.tracking.env_cfgs import dash_flat_tracking_env_cfg

# The CSV column order, and the contract between a retargeter and this script.
# Written out rather than read off the model so that reordering bodies in the
# MJCF cannot silently repermute existing CSVs; the names are resolved against
# the model at runtime, so a rename fails loudly instead.
CSV_JOINT_ORDER = (
  "r_hip_yaw",
  "r_hip_roll",
  "r_hip_pitch",
  "r_knee_pitch",
  "r_ankle_pitch",
  "l_hip_yaw",
  "l_hip_roll",
  "l_hip_pitch",
  "l_knee_pitch",
  "l_ankle_pitch",
  "r_shoulder_pitch",
  "r_shoulder_roll",
  "r_shoulder_yaw",
  "r_elbow_pitch",
  "l_shoulder_pitch",
  "l_shoulder_roll",
  "l_shoulder_yaw",
  "l_elbow_pitch",
)

_LOG_KEYS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


def main(
  input_file: str,
  output_file: str,
  input_fps: int = 30,
  output_fps: int = 50,
  device: str = "cuda:0",
  line_range: tuple[int, int] | None = None,
) -> None:
  """Replay a retargeted CSV through the Dash model and save the tracking NPZ.

  Args:
    input_file: Retargeted motion CSV (see the module docstring for columns).
    output_file: Where to write the NPZ.
    input_fps: Frame rate the CSV was sampled at.
    output_fps: Frame rate to resample to. Match the env's step rate.
    device: Torch device. Falls back to CPU when CUDA is unavailable.
    line_range: Optional 1-based inclusive row range, to clip a long take.
  """
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA is not available. Falling back to CPU. This may be slow.")
    device = "cpu"

  sim_cfg = SimulationCfg()
  # The replay writes each frame's state directly and only calls forward(), so
  # the timestep is not integrating anything -- it just has to agree with the
  # frame spacing for the velocities mjlab logs to come out in the right units.
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  scene = Scene(dash_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  robot = scene["robot"]
  joint_ids = robot.find_joints(list(CSV_JOINT_ORDER), preserve_order=True)[0]
  if len(joint_ids) != len(CSV_JOINT_ORDER):
    raise SystemExit(
      f"expected {len(CSV_JOINT_ORDER)} joints, resolved {len(joint_ids)}; "
      "CSV_JOINT_ORDER is out of step with the MJCF."
    )

  motion = MotionLoader(
    motion_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
    line_range=line_range,
  )
  n_csv_joints = motion.motion_dof_poss.shape[1]
  if n_csv_joints != len(CSV_JOINT_ORDER):
    raise SystemExit(
      f"{input_file} has {n_csv_joints} joint columns, Dash has "
      f"{len(CSV_JOINT_ORDER)}. Expected 7 base columns then "
      f"{len(CSV_JOINT_ORDER)} joint angles."
    )

  log: dict[str, Any] = {"fps": [output_fps]}
  for key in _LOG_KEYS:
    log[key] = []

  scene.reset()
  print(f"\nReplaying {motion.output_frames} frames at {output_fps} fps...")

  for _ in tqdm(range(motion.output_frames), desc="frames", ncols=88):
    (
      (
        base_pos,
        base_rot,
        base_lin_vel,
        base_ang_vel,
        dof_pos,
        dof_vel,
      ),
      _,
    ) = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = base_rot
    root_states[:, 7:10] = base_lin_vel
    root_states[:, 10:] = base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, joint_ids] = dof_pos
    joint_vel[:, joint_ids] = dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    # forward(), not step(): this is pure kinematics. The point is to read out
    # where every body ends up, not to simulate whether the motion is dynamically
    # feasible -- the policy's job is to find that out.
    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
    log["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
    log["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
    log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0].cpu().numpy().copy())
    log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0].cpu().numpy().copy())

    # The root body's state should be exactly what we wrote. If this trips, the
    # CSV's base columns are not landing on the root body and every body array
    # in the NPZ is anchored to the wrong frame.
    torch.testing.assert_close(robot.data.body_link_lin_vel_w[0, 0], base_lin_vel[0])
    torch.testing.assert_close(robot.data.body_link_ang_vel_w[0, 0], base_ang_vel[0])

  for key in _LOG_KEYS:
    log[key] = np.stack(log[key], axis=0)
  # Provenance: the NPZ is meaningless against a different model, and body
  # arrays are positional, so record what they were built from.
  log["joint_names"] = np.array(robot.joint_names)
  log["body_names"] = np.array(robot.body_names)

  out = Path(output_file)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez(out, **log)

  print(f"\nWrote {out}")
  print(f"  frames     : {log['joint_pos'].shape[0]} @ {output_fps} fps")
  print(f"  joints     : {log['joint_pos'].shape[1]}")
  print(f"  bodies     : {log['body_pos_w'].shape[1]}")
  print(
    f"  root height: {log['body_pos_w'][:, 0, 2].min():.3f} - "
    f"{log['body_pos_w'][:, 0, 2].max():.3f} m"
  )
  print("\nTrain with:")
  print("  dash-train Mjlab-Tracking-Flat-Dash \\")
  print(f"      --env.commands.motion.motion-file {out}")


def entry() -> None:
  tyro.cli(main, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
  entry()
