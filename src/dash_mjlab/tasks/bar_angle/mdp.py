"""MDP terms for the Dash bar-angle task.

The object here is not a free body but a 1-DOF mechanism: a horizontal bar on a
vertical hinge, whose entire state is one angle and one angular velocity. None
of mjlab's manipulation terms speak that language -- they all measure Cartesian
distances to a free object's root -- so the command, the observations, the
rewards and the terminations are all defined here in terms of the hinge angle.

Angles are the bar's hinge coordinate ``q`` (rad): ``q = 0`` is the spawn pose
with the bar pointing back at the robot, positive ``q`` swings the free end to
the robot's right (right-hand rule about +z). All angle arithmetic goes through
``wrap_to_pi`` so an error never takes the long way round the circle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, sample_uniform, wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


##
# Commands.
##


class BarAngleCommand(CommandTerm):
  """Command the bar to point at a target hinge angle, settled.

  At every episode start the bar is respawned at a random angle inside
  ``angle_range`` and a target is sampled at least ``min_separation`` away, so
  no episode begins already solved. Mid-episode resamples move only the target
  -- the bar stays exactly where the robot left it, so a resample reads as
  "now swing it somewhere else" rather than a teleport under the hands.

  Success is settle-and-hold: the wrapped angle error must be inside
  ``success_threshold`` *and* the bar's speed inside ``settle_speed``. A bar
  slapped through the target at speed does not count.

  The command the policy observes is ``(sin, cos)`` of the target angle, which
  is continuous where a raw angle would jump at the wrap.
  """

  cfg: BarAngleCommandCfg  # pyright: ignore[reportIncompatibleVariableOverride]

  def __init__(self, cfg: BarAngleCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.bar: Entity = env.scene[cfg.entity_name]
    self._joint_idx = self.bar.joint_names.index(cfg.joint_name)
    self._joint_ids = torch.tensor([self._joint_idx], device=self.device)

    self.target_angle = torch.zeros(self.num_envs, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)
    # Set for envs whose bar needs a fresh spawn angle, i.e. episode starts.
    self._respawn = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    self.metrics["angle_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return torch.stack(
      [torch.sin(self.target_angle), torch.cos(self.target_angle)], dim=-1
    )

  def bar_angle(self) -> torch.Tensor:
    return self.bar.data.joint_pos[:, self._joint_idx]

  def bar_speed(self) -> torch.Tensor:
    return self.bar.data.joint_vel[:, self._joint_idx]

  def angle_error(self) -> torch.Tensor:
    """Wrapped signed error target - current, in rad."""
    return wrap_to_pi(self.target_angle - self.bar_angle())

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    self._respawn[slice(None) if env_ids is None else env_ids] = True
    return super().reset(env_ids)

  def _update_metrics(self) -> None:
    error = torch.abs(self.angle_error())
    at_goal = (
      (error < self.cfg.success_threshold)
      & (self.bar_speed().abs() < self.cfg.settle_speed)
    ).float()
    self.episode_success = torch.maximum(self.episode_success, at_goal)
    self.metrics["angle_error"] = error
    self.metrics["at_goal"] = at_goal
    self.metrics["episode_success"] = self.episode_success

  def compute_success(self) -> torch.Tensor:
    return self.metrics["at_goal"] > 0.5

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    cfg = self.cfg
    self.episode_success[env_ids] = 0.0
    lo, hi = cfg.angle_range

    respawn = self._respawn[env_ids]
    spawn_angle = sample_uniform(lo, hi, (n,), device=self.device)
    current = torch.where(
      respawn, spawn_angle, self.bar.data.joint_pos[env_ids, self._joint_idx]
    )

    # Sample the target uniformly over the part of the range at least
    # min_separation from the current angle: the union of [lo, current - sep]
    # and [current + sep, hi], one draw split across the two by their lengths.
    # A rejection loop would do the same job but has no bounded iteration count.
    sep = cfg.min_separation
    below_len = (torch.clamp(current - sep, max=hi) - lo).clamp_min(0.0)
    above_start = torch.clamp(current + sep, min=lo)
    above_len = (hi - above_start).clamp_min(0.0)
    total = below_len + above_len
    u = torch.rand(n, device=self.device) * total
    target = torch.where(u < below_len, lo + u, above_start + (u - below_len))
    # Both intervals empty means the current angle blankets the whole range
    # (only possible if the range is narrower than 2*sep, or the bar was shoved
    # far outside it). Fall back to the range end furthest from the bar.
    fallback = torch.where(
      current < 0.5 * (lo + hi),
      torch.full((n,), hi, device=self.device),
      torch.full((n,), lo, device=self.device),
    )
    self.target_angle[env_ids] = torch.where(total > 0.0, target, fallback)

    respawn_ids = env_ids[respawn]
    if len(respawn_ids) > 0:
      pos = spawn_angle[respawn].unsqueeze(-1)
      self.bar.write_joint_state_to_sim(
        pos, torch.zeros_like(pos), joint_ids=self._joint_ids, env_ids=respawn_ids
      )
      self._respawn[respawn_ids] = False

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    px, py, pz = self.cfg.pivot_pos
    for batch in env_indices:
      origin = self._env.scene.env_origins[batch].cpu().numpy()
      angle = float(self.target_angle[batch])
      pivot = origin + (px, py, pz)
      # The bar geom points along local -x, so the tip of a bar at hinge angle
      # q sits at pivot + L * (-cos q, -sin q, 0).
      tip = pivot + self.cfg.bar_length * torch.tensor(
        [-math.cos(angle), -math.sin(angle), 0.0]
      ).numpy()
      visualizer.add_cylinder(
        start=pivot,
        end=tip,
        radius=0.005,
        color=self.cfg.viz_target_color,
        label=f"bar_target_{batch}",
      )
      visualizer.add_sphere(
        center=tip,
        radius=0.025,
        color=self.cfg.viz_target_color,
        label=f"bar_target_tip_{batch}",
      )


@dataclass(kw_only=True)
class BarAngleCommandCfg(CommandTermCfg):
  """Config for :class:`BarAngleCommand`. Angles in rad, hinge convention."""

  entity_name: str
  joint_name: str = "bar_joint"

  angle_range: tuple[float, float] = (-0.8, 0.8)
  """Both the spawn angle and the target are drawn from this interval."""

  min_separation: float = 0.5
  """Minimum wrapped distance between the bar's angle at resample time and the
  new target, so no command is trivially satisfied at issue."""

  success_threshold: float = 0.1
  """Max |angle error| that counts as at-goal (~6 degrees)."""

  settle_speed: float = 0.5
  """Max |bar speed| (rad/s) that counts as settled. Together with the damping
  this is what makes success mean "held there", not "swung through"."""

  pivot_pos: tuple[float, float, float] = (0.34, 0.0, 0.69)
  """Hinge position relative to the environment origin. Debug-vis only."""

  bar_length: float = 0.2
  """Pivot-to-tip length. Debug-vis only."""

  viz_target_color: tuple[float, float, float, float] = (0.1, 0.8, 0.2, 0.5)

  def build(self, env: ManagerBasedRlEnv) -> BarAngleCommand:
    return BarAngleCommand(self, env)


def _get_command(env: ManagerBasedRlEnv, command_name: str) -> BarAngleCommand:
  return cast(BarAngleCommand, env.command_manager.get_term(command_name))


##
# Observations.
##


def joint_angle_sincos(
  env: ManagerBasedRlEnv, object_name: str, joint_name: str
) -> torch.Tensor:
  """(sin, cos) of one joint of an entity. Shape (B, 2).

  Sin/cos rather than the raw angle so the observation is continuous through
  the +-pi wrap -- the bar itself cannot cross it inside the command range, but
  a shoved bar can, and a discontinuous observation there is a free bug.
  """
  obj: Entity = env.scene[object_name]
  q = obj.data.joint_pos[:, obj.joint_names.index(joint_name)]
  return torch.stack([torch.sin(q), torch.cos(q)], dim=-1)


def joint_velocity(
  env: ManagerBasedRlEnv, object_name: str, joint_name: str
) -> torch.Tensor:
  """Velocity of one joint of an entity. Shape (B, 1)."""
  obj: Entity = env.scene[object_name]
  return obj.data.joint_vel[:, [obj.joint_names.index(joint_name)]]


def bar_angle_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Signed wrapped error target - current. Shape (B, 1).

  Redundant with (bar angle, target) in principle, but handing the policy the
  subtraction it would otherwise have to learn is cheap and is the single most
  reward-relevant scalar in the task.
  """
  return _get_command(env, command_name).angle_error().unsqueeze(-1)


def hands_to_site_distance(
  env: ManagerBasedRlEnv,
  object_cfg: SceneEntityCfg,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Distance vector from each hand to a site on another entity, in the robot
  base frame. Shape (B, 6): right hand's vector then left hand's, in the order
  ``asset_cfg.site_names`` was given (needs ``preserve_order``).

  For the bar the site is the free tip -- the point of maximum leverage, and
  the part that actually traces the commanded arc.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_cfg.name]
  hands_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids]  # (B, 2, 3)
  site_pos_w = obj.data.site_pos_w[:, object_cfg.site_ids]  # (B, 1, 3)
  distance_vec_w = site_pos_w - hands_pos_w  # (B, 2, 3)
  num_hands = hands_pos_w.shape[1]
  base_quat_w = robot.data.root_link_quat_w.unsqueeze(1).expand(-1, num_hands, -1)
  distance_vec_b = quat_apply_inverse(base_quat_w, distance_vec_w)
  return distance_vec_b.flatten(start_dim=1)


##
# Rewards.
##


def bar_angle_tracking(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """exp(-error^2 / std^2) on the wrapped angle error.

  Used twice at different widths, mirroring the lift task's coarse/fine split:
  a wide kernel that has gradient across the whole command range, and a narrow
  one that only pays near the target and makes the last few degrees worth
  finishing.
  """
  error = _get_command(env, command_name).angle_error()
  return torch.exp(-torch.square(error) / std**2)


def bar_at_target(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """1.0 while the bar is at the target angle *and* settled, else 0.0.

  The discrete counterpart of the tracking kernels, and the term that pays for
  holding: a policy that bats the bar back and forth through the target
  collects it only in flashes, one that parks the bar collects it every step.
  Uses the same test as the command's success metric, so what is rewarded and
  what is reported as success cannot drift apart.
  """
  return _get_command(env, command_name).metrics["at_goal"]


def hands_to_site_reaching(
  env: ManagerBasedRlEnv,
  object_cfg: SceneEntityCfg,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """exp(-mean squared hand-to-site distance / std^2).

  Bootstrap term: before the first contact the angle kernels are flat (the bar
  only moves if something touches it), and this is the gradient that gets the
  hands to the bar at all. The mean over both hands sends them to straddle the
  tip, which leaves one hand on each side -- the position from which either
  push direction is available.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_cfg.name]
  hands_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids]  # (B, 2, 3)
  site_pos_w = obj.data.site_pos_w[:, object_cfg.site_ids]  # (B, 1, 3)
  reach_error = torch.sum(torch.square(hands_pos_w - site_pos_w), dim=-1).mean(dim=-1)
  return torch.exp(-reach_error / std**2)


##
# Terminations.
##


def bar_out_of_range(
  env: ManagerBasedRlEnv, object_name: str, joint_name: str, max_angle: float
) -> torch.Tensor:
  """True once |bar angle| exceeds ``max_angle``.

  Past ~85 degrees the bar lies along the robot's frontal plane at the arms'
  maximum forward reach, where neither hand can get a surface behind it to push
  it back -- the episode is dead time from there on, the same way a box off the
  table's edge was in the lift task.
  """
  obj: Entity = env.scene[object_name]
  q = obj.data.joint_pos[:, obj.joint_names.index(joint_name)]
  return q.abs() > max_angle


def unstable_state(
  env: ManagerBasedRlEnv,
  object_name: str,
  max_bar_speed: float,
  max_joint_speed: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """True when the simulation has diverged, or is about to.

  Same safety net as the lift task's, restated for a jointed object: the bar
  has no root velocity to check (its root is welded), so the divergence shows
  up in its hinge coordinate instead. The explicit finiteness test matters for
  the same reason as there -- every comparison against NaN is False, so the
  speed thresholds alone would never fire on the states this exists to catch.
  """
  obj: Entity = env.scene[object_name]
  robot: Entity = env.scene[asset_cfg.name]

  non_finite = (
    ~torch.isfinite(obj.data.joint_pos).all(dim=-1)
    | ~torch.isfinite(obj.data.joint_vel).all(dim=-1)
    | ~torch.isfinite(robot.data.joint_pos).all(dim=-1)
    | ~torch.isfinite(robot.data.joint_vel).all(dim=-1)
  )
  too_fast = (
    torch.nan_to_num(obj.data.joint_vel).abs().amax(dim=-1) > max_bar_speed
  ) | (torch.nan_to_num(robot.data.joint_vel).abs().amax(dim=-1) > max_joint_speed)
  return non_finite | too_fast
