"""Bimanual MDP terms for the Dash upper-body pick-and-place task.

mjlab's manipulation terms all assume a single end-effector site and a gripper
that can close around an object (see
``mjlab.tasks.manipulation.mdp.observations`` / ``.rewards``). Dash has neither:
two 4-DOF arms ending in bare rounded forearm tips, which hold a box by
squeezing it between them. The terms here are the two-handed counterparts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.manipulation.mdp.commands import LiftingCommand, LiftingCommandCfg
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  quat_from_euler_xyz,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


##
# Commands.
##


class TraverseLiftingCommand(LiftingCommand):
  """Pick at one end of the reachable strip, place at the other.

  mjlab's ``LiftingCommand`` samples the object pose and the goal independently
  from two overlapping boxes, so a typical episode asks for a short hop -- and
  some episodes ask for no travel at all, because the goal lands on top of where
  the box already is. This term samples the two as a matched pair on opposite
  ends instead, so every episode is a full traverse.

  Both the end that is picked from and the lateral side are randomized, so the
  policy sees near-to-far and far-to-near, left-to-right and right-to-left,
  rather than memorizing one trajectory.

  Everything else -- metrics, success, the goal marker -- comes from the parent.
  ``object_pose_range`` and ``target_position_range`` on the config are unused
  here; the ranges below replace them.
  """

  # Narrowing the base class's `cfg` to the subclass config, the same pattern
  # mjlab's own command terms use to reach their extra fields.
  cfg: TraverseLiftingCommandCfg  # pyright: ignore[reportIncompatibleVariableOverride]

  def __init__(self, cfg: TraverseLiftingCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.metrics["traverse_distance"] = torch.zeros(self.num_envs, device=self.device)
    # Set for envs whose box needs putting back on the table, i.e. episode
    # starts. See _resample_command.
    self._respawn = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    # A new episode is the only time the box's position carries no information
    # -- it may be anywhere, including on the floor -- so it is the only time
    # the box gets moved.
    self._respawn[slice(None) if env_ids is None else env_ids] = True
    return super().reset(env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self.episode_success[env_ids] = 0.0
    cfg = self.cfg

    def uniform(rng: tuple[float, float]) -> torch.Tensor:
      return sample_uniform(rng[0], rng[1], (n,), device=self.device)

    origins = self._env.scene.env_origins[env_ids]
    respawn = self._respawn[env_ids]

    # Where the box is, or is about to be put. Mid-episode the box is left
    # exactly where it is: teleporting it is not a neutral act when the robot
    # is holding it, and by the time the policy can grasp, it usually is. The
    # box lands inside two squeezing hands, which drives a deep interpenetration
    # and a force spike measured at 217 N against a normal grip of 30-75 N. It
    # also destroys the credit for the grasp that was in progress. Only the goal
    # moves, so a resample becomes "now take it back the other way".
    side = torch.randint(0, 2, (n,), device=self.device) * 2.0 - 1.0
    pick_x = torch.where(
      torch.randint(0, 2, (n,), device=self.device) > 0,
      uniform(cfg.near_x),
      uniform(cfg.far_x),
    )
    pick_pos = (
      torch.stack(
        [
          pick_x,
          side * uniform(cfg.lateral_y),
          torch.full((n,), cfg.surface_z, device=self.device),
        ],
        dim=-1,
      )
      + origins
    )

    box_pos = self.object.data.root_link_pos_w[env_ids]
    box_pos = torch.where(respawn.unsqueeze(-1), pick_pos, box_pos)

    # The goal is always the opposite corner from wherever the box is: the far
    # zone if it is nearer than the midpoint, the near zone if it is further,
    # and the mirrored lateral offset either way.
    local = box_pos - origins
    mid_x = 0.5 * (sum(cfg.near_x) + sum(cfg.far_x)) * 0.5
    goal_x = torch.where(local[:, 0] < mid_x, uniform(cfg.far_x), uniform(cfg.near_x))
    # sign() is 0 for a box exactly on the centreline, which would put the goal
    # there too; treat that as "came from the right".
    box_side = torch.where(local[:, 1] >= 0.0, 1.0, -1.0)
    goal_y = -box_side * uniform(cfg.lateral_y)
    place_pos = (
      torch.stack(
        [goal_x, goal_y, torch.full((n,), cfg.surface_z, device=self.device)], dim=-1
      )
      + origins
    )

    self.target_pos[env_ids] = place_pos
    self.metrics["traverse_distance"][env_ids] = torch.norm(place_pos - box_pos, dim=-1)

    respawn_ids = env_ids[respawn]
    if len(respawn_ids) > 0:
      zeros = torch.zeros(len(respawn_ids), device=self.device)
      quat = quat_from_euler_xyz(
        zeros,
        zeros,
        sample_uniform(
          cfg.yaw_range[0], cfg.yaw_range[1], (len(respawn_ids),), device=self.device
        ),
      )
      self.object.write_root_link_pose_to_sim(
        torch.cat([pick_pos[respawn], quat], dim=-1), env_ids=respawn_ids
      )
      self.object.write_root_link_velocity_to_sim(
        torch.zeros(len(respawn_ids), 6, device=self.device), env_ids=respawn_ids
      )
      self._respawn[respawn_ids] = False


@dataclass(kw_only=True)
class TraverseLiftingCommandCfg(LiftingCommandCfg):
  """Config for :class:`TraverseLiftingCommand`.

  All positions are relative to the environment origin, except ``surface_z``
  which is an absolute height.
  """

  near_x: tuple[float, float] = (0.23, 0.25)
  """Near end of the traverse, measured from the torso."""

  far_x: tuple[float, float] = (0.30, 0.32)
  """Far end. Both ends sit inside the strip where a squeeze actually grips,
  measured by simulation as x in [0.22, 0.32] at table height."""

  lateral_y: tuple[float, float] = (0.04, 0.06)
  """Magnitude of the lateral offset. Pick and place take opposite signs, so
  the traverse is a diagonal across the strip. The lateral axis is the longer
  of the two, which is why it carries the larger share of the distance."""

  surface_z: float = 0.69
  """Height of the box centre resting on the table. Both the pick and the place
  are on the surface: this is a place, not a hold-it-in-the-air."""

  yaw_range: tuple[float, float] = (-0.3, 0.3)
  """Spawn yaw of the box."""

  def build(self, env: ManagerBasedRlEnv) -> TraverseLiftingCommand:
    return TraverseLiftingCommand(self, env)


def _hand_positions(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Both hand site positions in world frame. Shape (B, 2, 3)."""
  robot: Entity = env.scene[asset_cfg.name]
  return robot.data.site_pos_w[:, asset_cfg.site_ids]


##
# Observations.
##


def hands_to_object_distance(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Distance vector from each hand to the object, in the robot base frame.

  Shape (B, 6): the right hand's vector followed by the left hand's, in the
  order ``asset_cfg.site_names`` was given (which needs ``preserve_order``).
  Two separate vectors rather than one to their midpoint: the midpoint is
  identical whether the hands straddle the box or are both sitting off to one
  side of it, and only the first of those is a grasp.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  hands_pos_w = _hand_positions(env, asset_cfg)  # (B, 2, 3)
  obj_pos_w = obj.data.root_link_pos_w  # (B, 3)
  distance_vec_w = obj_pos_w.unsqueeze(1) - hands_pos_w  # (B, 2, 3)
  num_hands = hands_pos_w.shape[1]
  base_quat_w = robot.data.root_link_quat_w.unsqueeze(1).expand(-1, num_hands, -1)
  distance_vec_b = quat_apply_inverse(base_quat_w, distance_vec_w)
  return distance_vec_b.flatten(start_dim=1)


def hand_separation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Distance between the two hand sites. Shape (B, 1).

  This is what the policy has to modulate to grip: wide enough to get around
  the box, then narrower than the box to press on it.
  """
  hands_pos_w = _hand_positions(env, asset_cfg)
  return torch.norm(hands_pos_w[:, 0] - hands_pos_w[:, 1], dim=-1, keepdim=True)


##
# Rewards.
##


def bimanual_staged_position_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  reaching_std: float,
  bringing_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Two-handed version of mjlab's ``staged_position_reward``.

  Returns ``reaching * (1 + bringing)``, so bringing the box to the goal only
  pays while both hands are still on it -- the policy cannot farm the bringing
  term by throwing the box goalward and letting go.

  Reaching drives *both* hands toward the box centre, which is the whole grasp:
  the hands cannot pass through the box, so the only way to keep shrinking both
  distances at once is to close on it from opposite sides. The mean over the two
  hands (not the min) means a single hand parked on the box earns roughly half,
  never the whole term.
  """
  obj: Entity = env.scene[object_name]
  command = cast(LiftingCommand, env.command_manager.get_term(command_name))

  hands_pos_w = _hand_positions(env, asset_cfg)  # (B, 2, 3)
  obj_pos_w = obj.data.root_link_pos_w  # (B, 3)

  reach_error = torch.sum(
    torch.square(hands_pos_w - obj_pos_w.unsqueeze(1)), dim=-1
  ).mean(dim=-1)
  reaching = torch.exp(-reach_error / reaching_std**2)

  position_error = torch.sum(torch.square(command.target_pos - obj_pos_w), dim=-1)
  bringing = torch.exp(-position_error / bringing_std**2)

  return reaching * (1.0 + bringing)


def both_hands_in_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 1.0,
) -> torch.Tensor:
  """1.0 while both hands press on the object, else 0.0.

  The staged reward above is a smooth distance kernel, which keeps paying while
  the hands merely hover near the box. This term is the discrete "you are
  actually squeezing it" signal, and it is what makes the difference between
  nudging the box across the table and picking it up.

  ``force_threshold`` filters grazes: a real squeeze on a 0.25 kg box needs
  several newtons per pad, so 1 N is comfortably below a genuine grip and above
  contact noise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  assert data.force is not None, (
    f"Contact sensor '{sensor_name}' must request the 'force' field."
  )
  # force: (B, N, 3) with one slot per primary geom, i.e. one per hand.
  force_mag = torch.norm(data.force, dim=-1)  # (B, N)
  return (force_mag > force_threshold).all(dim=-1).float()


def object_height_above(
  env: ManagerBasedRlEnv,
  object_name: str,
  reference_height: float,
) -> torch.Tensor:
  """Height of the object above ``reference_height``, clamped at zero.

  Measured against the environment origin, so it is the same number in every
  environment of the grid. Pass the table's top surface as the reference: this
  then rewards clearing the table at all, which is the step the distance kernels
  are worst at teaching (a box slid to below the goal scores well on horizontal
  distance and never leaves the surface).
  """
  obj: Entity = env.scene[object_name]
  height = obj.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return (height - reference_height).clamp_min(0.0)


##
# Terminations.
##


def unstable_state(
  env: ManagerBasedRlEnv,
  object_name: str,
  max_object_speed: float,
  max_joint_speed: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """True when the simulation has diverged, or is about to.

  A safety net, not a task rule. A squeeze grip is a stiff contact problem, and
  a rare solver failure sends a velocity to infinity; the NaN then reaches the
  observations and kills the training run outright, hours in. Terminating the
  affected environment lets the reset write finite state back and training
  carries on, losing one episode instead of the run.

  The finiteness check has to be explicit. Every comparison against NaN is
  False, so a plain speed threshold silently fails to fire on exactly the states
  it is meant to catch.
  """
  obj: Entity = env.scene[object_name]
  robot: Entity = env.scene[asset_cfg.name]

  obj_vel = obj.data.root_link_lin_vel_w
  joint_vel = robot.data.joint_vel

  non_finite = (
    ~torch.isfinite(obj.data.root_link_pos_w).all(dim=-1)
    | ~torch.isfinite(obj_vel).all(dim=-1)
    | ~torch.isfinite(robot.data.joint_pos).all(dim=-1)
    | ~torch.isfinite(joint_vel).all(dim=-1)
  )
  too_fast = (torch.nan_to_num(obj_vel).norm(dim=-1) > max_object_speed) | (
    torch.nan_to_num(joint_vel).abs().amax(dim=-1) > max_joint_speed
  )
  return non_finite | too_fast


def object_dropped(
  env: ManagerBasedRlEnv,
  object_name: str,
  minimum_height: float,
) -> torch.Tensor:
  """True once the object falls below ``minimum_height`` above the env origin.

  A box knocked off the table cannot be recovered -- the arms cannot reach the
  floor -- so the rest of that episode is dead time. Set the threshold below the
  table surface but above the floor.
  """
  obj: Entity = env.scene[object_name]
  height = obj.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return height < minimum_height
