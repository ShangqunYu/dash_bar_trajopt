"""Dash flat motion tracking environment config.

NOTE: this task needs retargeted reference motion for Dash. mjlab's tracking
task loads a motion file at train time (``--registry-name`` / the motion command
config); none exists for Dash yet, so this config registers and constructs but
will not train. See mjlab's motion imitation docs for the
retargeting pipeline.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from dash_mjlab.robots import DASH_ACTION_SCALE, get_dash_robot_cfg


def dash_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the Dash flat terrain motion tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_dash_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = DASH_ACTION_SCALE

  # Dash has no separate pelvis link, so the torso serves as both the anchor and
  # the root body of the tracked chain.
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso"
  motion_cmd.body_names = (
    "torso",
    "l_dist_hip",
    "l_lower_leg",
    "l_foot",
    "r_dist_hip",
    "r_lower_leg",
    "r_foot",
    "l_dist_shoulder",
    "l_lower_arm",
    "r_dist_shoulder",
    "r_lower_arm",
  )

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = r"^[lr]_foot_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "l_foot",
    "r_foot",
    "l_lower_arm",
    "r_lower_arm",
  )

  cfg.viewer.body_name = "torso"

  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg
