"""Dash upper-body bar-angle environment configuration.

The object is a horizontal bar on a vertical revolute joint -- a 1-DOF
mechanism, not a free body. The hinge axis is parallel to gravity, so gravity
exerts zero torque about it: the bar never falls, it stays at whatever angle it
is left at (minus what the joint damping bleeds off). The task is to push the
bar so it points at a commanded angle and hold it there.

Like the lift-box task this is built on mjlab's single-arm lift-cube config,
keeping its action term, penalty rewards, joint-velocity curriculum and solver
settings, and replacing the object, the command, the observations, the rewards
and the terminations.

Scene geometry, all relative to the torso at x = y = 0:

  z = 0.69    bar swing plane -- hand height, same as the box task's grip height
  x = 0.34    pivot, one centimetre past the arms' measured forward reach
  x = 0.14    bar tip at spawn (q = 0), pointing back at the robot

The pivot sits just *outside* the workspace on purpose: the bar then crosses
the reachable strip (x in [0.22, 0.32] at this height, measured for the box
task) at every angle in the command range, so some part of it is always
pushable, while the post itself stays out of the arms' way. The tip at spawn
clears the torso's collision box (which ends at x = 0.08) by 6 cm.
"""

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from dash_mjlab.robots import (
  DASH_HAND_GEOMS,
  DASH_HAND_SITES,
  DASH_UPPER_BODY_ACTION_SCALE,
  get_dash_upper_body_robot_cfg,
)
from dash_mjlab.tasks.bar_angle import mdp

##
# Scene geometry.
##

BAR_HEIGHT = 0.69
"""Height of the bar's swing plane. The height the hands sit at in the spawn
pose and the height the box task gripped at -- the arms' sweet spot."""

PIVOT_X = 0.34
"""Pivot distance in front of the torso. Just past the arms' forward reach
(the box task measured contact possible up to x = 0.32, impossible at 0.34),
so the arms cannot jam themselves against the post, while every point of the
bar inboard of the pivot stays closer than the pivot itself."""

BAR_LENGTH = 0.20
"""Pivot-to-tip length. With the pivot at 0.34 the inner half of the bar
sweeps through the strip the arms are known to reach at every command angle;
the tip at spawn sits at x = 0.14, clear of the torso box at 0.08."""

BAR_RADIUS = 0.02
BAR_MASS = 0.3

BAR_DAMPING = 0.05
"""Hinge damping (Nm s/rad). Sized against the bar's inertia about the hinge,
m L^2 / 3 = 4e-3 kg m^2: the time constant I/d is ~0.1 s, so the bar stops
about where the hand leaves it instead of coasting -- which is what makes
"hold it at the angle" a matter of stopping in the right place rather than of
fighting a flywheel. The torque this costs the robot at working speeds is
~0.1 Nm against a 30 Nm effort limit. Randomized 0.5x-2x per run (see the
bar_damping event) so the policy cannot tune to one resistance."""

POST_RADIUS = 0.02

ANGLE_RANGE = (-0.8, 0.8)
"""Spawn and target angles (rad, ~+-46 deg). Limited arc: across this range
the bar always crosses the arms' reachable strip, so every commanded angle is
attainable with a push somewhere along the bar."""

MIN_SEPARATION = 0.5
"""Minimum spawn-to-target (and resample-to-target) angular distance, so every
command demands a swing of at least ~29 deg."""

SUCCESS_THRESHOLD = 0.1
"""At-goal angle tolerance (~6 deg)."""

SETTLE_SPEED = 0.5
"""At-goal bar speed tolerance (rad/s). At this speed the damping alone parks
the bar within the angle tolerance, so "at goal and slower than this" really
is settled."""

BAR_LIMIT_ANGLE = 1.5
"""|angle| that terminates the episode (~86 deg). Past it the bar lies along
the frontal plane at maximum reach where no hand can get behind it, so the
episode is unrecoverable -- the analogue of the box off the table's edge."""


##
# Entity spec.
##


def get_bar_spec() -> mujoco.MjSpec:
  """A bar on a vertical hinge atop a floor-standing post.

  The post is the only static collider; the bar body carries the hinge and the
  capsule. Parent-child contacts are excluded by MuJoCo, so bar and post never
  collide with each other, and nothing here reaches the floor plane.
  """
  spec = mujoco.MjSpec()
  base = spec.worldbody.add_body(name="bar_base")
  # The post stops 3.5 cm short of the swing plane so the bar capsule
  # (radius 2 cm) clears it; a thin visual axle bridges the gap.
  base.add_geom(
    name="post",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    fromto=[0.0, 0.0, 0.0, 0.0, 0.0, BAR_HEIGHT - 0.035],
    size=[POST_RADIUS, 0.0, 0.0],
    rgba=(0.30, 0.30, 0.32, 1.0),
    # Frictionless: the post exists to be a physical obstacle, not a surface
    # the task ever wants the arms to work against.
    condim=1,
  )
  base.add_geom(
    name="post_axle_visual",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    fromto=[0.0, 0.0, BAR_HEIGHT - 0.035, 0.0, 0.0, BAR_HEIGHT + 0.01],
    size=[0.008, 0.0, 0.0],
    rgba=(0.55, 0.55, 0.58, 1.0),
    group=2,
    contype=0,
    conaffinity=0,
    density=0.0,
  )
  base.add_geom(
    name="post_foot_visual",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    fromto=[0.0, 0.0, 0.0, 0.0, 0.0, 0.012],
    size=[0.07, 0.0, 0.0],
    rgba=(0.24, 0.24, 0.26, 1.0),
    group=2,
    contype=0,
    conaffinity=0,
    density=0.0,
  )

  bar = base.add_body(name="bar", pos=[0.0, 0.0, BAR_HEIGHT])
  bar.add_joint(
    name="bar_joint",
    type=mujoco.mjtJoint.mjJNT_HINGE,
    axis=[0.0, 0.0, 1.0],
    damping=BAR_DAMPING,
    # A touch of reflected inertia keeps the hinge well-conditioned when a
    # stiff hand contact lands right at the pivot end.
    armature=0.001,
  )
  # Along local -x, so q = 0 points the bar back at the robot and the whole
  # command range keeps it in front of the pivot. Signal red: the bar's
  # direction is the entire task state, and it should be readable at a glance
  # against the grey post and floor.
  bar.add_geom(
    name="bar_geom",
    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    fromto=[0.0, 0.0, 0.0, -BAR_LENGTH, 0.0, 0.0],
    size=[BAR_RADIUS, 0.0, 0.0],
    mass=BAR_MASS,
    rgba=(0.82, 0.16, 0.12, 1.0),
    # As with the box: the hands' geoms carry priority 2, so on a hand-bar
    # contact the hand's friction wins outright and these values are moot.
    condim=3,
    friction=(0.6, 0.005, 0.0001),
  )
  bar.add_geom(
    name="bar_tip_visual",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    pos=[-BAR_LENGTH, 0.0, 0.0],
    size=[0.026, 0.0, 0.0],
    rgba=(0.95, 0.80, 0.15, 1.0),
    group=2,
    contype=0,
    conaffinity=0,
    density=0.0,
  )
  # The point the reaching terms measure to: the free end, maximum leverage.
  bar.add_site(name="bar_tip", pos=[-BAR_LENGTH, 0.0, 0.0], size=[0.005] * 3)
  return spec


##
# Environment config.
##


def dash_bar_angle_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Dash upper-body bar-angle configuration."""
  cfg = make_lift_cube_env_cfg()

  cfg.scene.entities = {
    "robot": get_dash_upper_body_robot_cfg(),
    "bar": EntityCfg(
      spec_fn=get_bar_spec,
      init_state=EntityCfg.InitialStateCfg(pos=(PIVOT_X, 0.0, 0.0)),
    ),
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = DASH_UPPER_BODY_ACTION_SCALE

  # preserve_order so site_ids follow DASH_HAND_SITES (right, then left) rather
  # than the model's declaration order. The observation layout depends on it.
  hands_cfg = SceneEntityCfg("robot", site_names=DASH_HAND_SITES, preserve_order=True)
  bar_tip_cfg = SceneEntityCfg("bar", site_names=("bar_tip",))

  ##
  # Observations.
  ##

  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=velocity_mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=velocity_mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "hands_to_bar_tip": ObservationTermCfg(
      func=mdp.hands_to_site_distance,
      params={"object_cfg": bar_tip_cfg, "asset_cfg": hands_cfg},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "bar_angle": ObservationTermCfg(
      func=mdp.joint_angle_sincos,
      params={"object_name": "bar", "joint_name": "bar_joint"},
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
    "bar_vel": ObservationTermCfg(
      func=mdp.joint_velocity,
      params={"object_name": "bar", "joint_name": "bar_joint"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "target_angle": ObservationTermCfg(
      func=base_mdp.generated_commands,
      params={"command_name": "bar_target"},
    ),
    "angle_error": ObservationTermCfg(
      func=mdp.bar_angle_error,
      params={"command_name": "bar_target"},
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
    "actions": ObservationTermCfg(func=velocity_mdp.last_action),
  }
  cfg.observations = {
    "actor": ObservationGroupCfg(dict(actor_terms), enable_corruption=True),
    "critic": ObservationGroupCfg(dict(actor_terms), enable_corruption=False),
  }

  ##
  # Commands.
  ##

  # The bar's spawn angle is written by the command term, not a reset event,
  # for the same reason the box task did it: spawn and target have to be
  # sampled together to enforce the minimum separation, and the command term
  # is the only place that sees both.
  cfg.commands = {
    "bar_target": mdp.BarAngleCommandCfg(
      entity_name="bar",
      resampling_time_range=(6.0, 10.0),
      debug_vis=True,
      angle_range=ANGLE_RANGE,
      min_separation=MIN_SEPARATION,
      success_threshold=SUCCESS_THRESHOLD,
      settle_speed=SETTLE_SPEED,
      pivot_pos=(PIVOT_X, 0.0, BAR_HEIGHT),
      bar_length=BAR_LENGTH,
    )
  }

  ##
  # Sensors.
  ##

  # The stock task terminates on end-effector/floor contact; these hands are
  # 0.6 m above the floor and cannot touch it. No replacement sensor: the task
  # needs no contact signal, since "the bar moved" is readable from its angle.
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "ee_ground_collision"
  )

  ##
  # Events.
  ##

  # Position the (fixed-base, hence mocap-wrapped) robot and bar post at their
  # per-environment origins. Without these they all stack at the world origin.
  cfg.events["reset_base"].params["asset_cfg"] = SceneEntityCfg("robot")
  cfg.events["reset_bar_base"] = EventTermCfg(
    func=velocity_mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {},
      "velocity_range": {},
      "asset_cfg": SceneEntityCfg("bar"),
    },
  )

  # Joint scatter at reset. Wider than the box task's 0.03: there is no spawn
  # clearance to protect here -- the bar at any spawn angle passes no closer
  # than ~0.14 m to a hand -- and more varied start poses cost nothing.
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)

  # Retarget the friction randomization from YAM's fingertips to the forearm
  # tips, and rename to match. The stock (0.3, 1.5) slide range stays: a push
  # works at any friction (unlike the box task's squeeze grip, which is
  # friction-critical), so even the slippery end of the range trains fine.
  for axis in ("slide", "spin", "roll"):
    term = cfg.events.pop(f"fingertip_friction_{axis}")
    term.params["asset_cfg"] = SceneEntityCfg("robot", geom_names=DASH_HAND_GEOMS)
    cfg.events[f"hand_friction_{axis}"] = term

  # Per-run scatter on the hinge damping, 0.5x-2x the nominal. Damping is the
  # only passive resistance the bar has, so this is the whole of the object's
  # dynamics randomization.
  cfg.events["bar_damping"] = EventTermCfg(
    func=dr.joint_damping,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("bar", joint_names=("bar_joint",)),
      "operation": "scale",
      "distribution": "log_uniform",
      "ranges": (0.5, 2.0),
    },
  )

  ##
  # Rewards.
  ##

  # The stock lift terms measure Cartesian distance to a free object's root,
  # which for a hinged bar is meaningless (its root never moves).
  cfg.rewards.pop("lift")
  cfg.rewards.pop("lift_precise")

  # Coarse/fine pair on the wrapped angle error, mirroring the stock task's
  # lift / lift_precise split: the wide kernel has gradient across the whole
  # +-0.8 rad command range, the narrow one makes the last ~6 degrees worth
  # finishing rather than hovering nearby.
  cfg.rewards["track_angle"] = RewardTermCfg(
    func=mdp.bar_angle_tracking,
    weight=1.0,
    params={"command_name": "bar_target", "std": 0.7},
  )
  cfg.rewards["track_angle_fine"] = RewardTermCfg(
    func=mdp.bar_angle_tracking,
    weight=1.0,
    params={"command_name": "bar_target", "std": 0.15},
  )
  # Bootstrap: before the first contact the angle kernels are flat, and this
  # is the only gradient pointing the hands at the bar. Small weight so it
  # never competes with tracking once contact is routine.
  cfg.rewards["reach_bar"] = RewardTermCfg(
    func=mdp.hands_to_site_reaching,
    weight=0.3,
    params={"object_cfg": bar_tip_cfg, "asset_cfg": hands_cfg, "std": 0.15},
  )
  # The settle-and-hold term: pays every step the bar is at the target angle
  # and slow, i.e. exactly the command's definition of success. This is what
  # separates parking the bar from batting it back and forth through the goal.
  cfg.rewards["settled"] = RewardTermCfg(
    func=mdp.bar_at_target,
    weight=1.0,
    params={"command_name": "bar_target"},
  )

  ##
  # Terminations.
  ##

  cfg.terminations.pop("ee_ground_collision", None)
  # Safety net against solver divergence, as in the box task. The thresholds
  # are far above anything the task produces: the bar peaks around 3 rad/s
  # under a hard shove, the arms near 6 rad/s under random actions.
  cfg.terminations["unstable"] = TerminationTermCfg(
    func=mdp.unstable_state,
    params={
      "object_name": "bar",
      "max_bar_speed": 50.0,
      "max_joint_speed": 100.0,
    },
  )
  cfg.terminations["bar_out_of_range"] = TerminationTermCfg(
    func=mdp.bar_out_of_range,
    params={
      "object_name": "bar",
      "joint_name": "bar_joint",
      "max_angle": BAR_LIMIT_ANGLE,
    },
  )

  cfg.viewer.body_name = "torso"
  cfg.viewer.distance = 1.6
  cfg.viewer.elevation = -30.0

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    # Resample more often so a play session shows many attempts.
    cfg.commands["bar_target"].resampling_time_range = (5.0, 5.0)

  return cfg
