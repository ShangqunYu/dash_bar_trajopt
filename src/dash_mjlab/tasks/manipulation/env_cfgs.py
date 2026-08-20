"""Dash upper-body bimanual pick-and-place environment configuration.

Built on mjlab's single-arm lift-cube task
(:func:`mjlab.tasks.manipulation.lift_cube_env_cfg.make_lift_cube_env_cfg`),
which supplies the action term, the penalty rewards, the joint-velocity
curriculum and the solver settings. Everything that assumes one arm and a
gripper -- the observations, the reach reward, the end-effector ground contact
termination -- is replaced here.

Scene geometry. The heights are chosen relative to the torso mount height in
``dash_upper_body_constants`` and validated against the compiled arms, so moving
the torso means re-running those checks:

  z = 0.688   torso origin (the height it sits at on the standing robot)
  z = 0.69    box centre at rest -- level with the hands in the spawn pose
  z = 0.63    table top
  z = 0.0     floor

The box is picked from one end of the arms' reachable strip and placed on the
other -- see PICK_X / PLACE_X / LATERAL_Y below, and the note there about why
the traverse runs fore-aft rather than left-right.
"""

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from dash_mjlab.robots import (
  DASH_HAND_GEOMS,
  DASH_HAND_SITES,
  DASH_UPPER_BODY_ACTION_SCALE,
  get_dash_upper_body_robot_cfg,
)
from dash_mjlab.tasks.manipulation import mdp

##
# Scene geometry.
##

TABLE_HEIGHT = 0.63
"""Height of the table's top surface. Chosen so that a box resting on it sits at
0.69 -- level with where the hands already are in the robot's spawn pose, and
the height at which the arms reach furthest forward. Dropping it to a more
conventional 0.60 costs ~4 cm of forward reach at the far edge of the box spawn
region, because the arms are sweeping an arc and the low end of that arc is
tucked back under the shoulder."""

TABLE_SIZE = (0.30, 0.46, 0.04)
"""Table top (depth, width, thickness) in metres. Sized to the arms rather than
to a person: the strip the robot can actually grip a box over is 10 cm deep, so
a full-size desk would leave it working a small patch in the middle and the
"carry it across the table" framing would be a fiction."""

TABLE_CENTER_X = 0.28
"""Table centre, in front of the robot. Spans x in [0.13, 0.43], which brackets
the 0.24-0.32 grip strip with room for the box's own footprint."""

BOX_HALF_EXTENTS = (0.05, 0.10, 0.06)
"""Half-extents of the box. The y extent is the one that matters: it sets how
far apart the two hands have to be to hold it, and therefore where in front of
the robot a grip is possible at all.

Too narrow and the arms cannot adduct that far -- shoulder roll runs out at
0.3 rad and the servos sag past it. Too wide and the far end of the workspace
goes out of reach. Simulating the squeeze across a grid gives a usable strip of
x in [0.24, 0.32] at 16 cm wide, against x in [0.22, 0.32] at 20 cm: widening
the box moves the *near* limit in, because holding the hands further apart is
the easy direction for these shoulders, while the far limit is the arm simply
running out of forward reach and does not move. 20 cm buys about 4 cm of
traverse for a parcel that is still narrower than the 22 cm shoulder span.

The 12 cm height matters less now that the contact is a sphere rather than a
flat pad, but it still keeps the contact point well clear of the table.
"""

BOX_MASS = 0.3
"""Light enough that a squeeze grip at the hands' randomized friction (0.8-1.5
slide) holds it: the two pads have to generate mu * F_normal > m * g between
them, i.e. ~2 N of squeeze at the worst-case friction. The scripted grasp check
produces ~20 N, so there is an order of magnitude of margin."""

# The traverse. All x/y are relative to the environment origin -- the robot's
# torso is at x = y = 0 -- and the box is both picked and placed on the table
# surface, so the z is BOX_REST_Z at each end.
#
# These come from simulating the grip, not from checking reach. Driving the
# hands hard into the box at each point of a grid and reading the contact force
# gives the strip x in [0.22, 0.32] at table height, y in [-0.06, 0.06]. The
# pick range starts a centimetre inside that: at x = 0.22 the box can be
# gripped on the table but not once lifted, and the carry needs both.  Outside
# it the grip does not merely weaken, it fails: at x = 0.34 the hands never
# touch the box at all. The strip is wider once the box is off the surface, so
# it is the pick and the place that bind, not the carry.
#
# The far limit is the forearm running out of reach; the near limit is torque.
# IK solves poses closer to the torso exactly, but shoulder roll is already
# pinned at its 0.3 rad limit there and the servo still settles wider than
# commanded under gravity alone.
#
# This strip sits ~4 cm closer to the robot than the flat-pad version did,
# because the contact point moved: the forearm's rounded cap is 4.5 cm further
# up the arm than a pad on its end would be, so the arm has to extend that much
# further to put the contact at the box's mid-height.
#
# The traverse runs diagonally across the strip rather than along one axis --
# 10 cm of depth crossed with 12 cm of width -- because neither axis alone is
# long enough to be worth calling a traverse. Pick and place are sampled at
# opposite corners (see TraverseLiftingCommand), so every episode travels
# 10-16 cm.
PICK_X = (0.23, 0.25)
PLACE_X = (0.30, 0.32)
LATERAL_Y = (0.04, 0.06)

BOX_REST_Z = TABLE_HEIGHT + BOX_HALF_EXTENTS[2]
"""Height of the box centre when it is sitting on the table -- the height of
both the pick and the place, since this is a place and not a hold-in-the-air."""

##
# Entity specs.
##


def _add_material(
  spec: mujoco.MjSpec,
  name: str,
  rgb1: tuple[float, float, float],
  rgb2: tuple[float, float, float],
  markrgb: tuple[float, float, float],
  texrepeat: tuple[float, float],
  texuniform: bool,
  specular: float,
  shininess: float,
) -> None:
  """Attach a procedural checker material to ``spec``.

  Builtin textures rather than image files: the whole scene stays a few lines of
  Python with no assets to ship, and the ground plane mjlab supplies is textured,
  so flat untextured rgba on the table and box reads as plastic beside it.
  """
  spec.add_texture(
    name=name,
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    mark=mujoco.mjtMark.mjMARK_EDGE,
    rgb1=list(rgb1),
    rgb2=list(rgb2),
    markrgb=list(markrgb),
    width=300,
    height=300,
  )
  material = spec.add_material(
    name=name,
    texrepeat=list(texrepeat),
    texuniform=texuniform,
    specular=specular,
    shininess=shininess,
    reflectance=0.0,
  )
  material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name


def get_box_spec() -> mujoco.MjSpec:
  """A single free box, the object to be picked and placed.

  Textured as a cardboard parcel: one texture tile per face with a darker edge
  mark, which reads as taped seams and -- more usefully -- makes the box's
  orientation legible at a glance, since the yaw it spawns with is something the
  policy has to cope with.
  """
  spec = mujoco.MjSpec()
  _add_material(
    spec,
    "cardboard",
    rgb1=(0.69, 0.52, 0.34),
    rgb2=(0.64, 0.47, 0.30),
    markrgb=(0.44, 0.32, 0.20),
    # One tile per face, not scaled by world size, so the edge mark lands on the
    # box's actual edges.
    texrepeat=(1.0, 1.0),
    texuniform=False,
    specular=0.1,
    shininess=0.1,
  )
  body = spec.worldbody.add_body(name="box")
  body.add_freejoint(name="box_joint")
  body.add_geom(
    name="box_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=BOX_HALF_EXTENTS,
    mass=BOX_MASS,
    material="cardboard",
    # condim 3 and no priority: the forearm tips carry priority 2, so on a
    # hand-box contact the hand's friction wins outright. This value only
    # governs the box sliding on the table.
    condim=3,
    friction=(0.6, 0.005, 0.0001),
  )
  return spec


def get_table_spec() -> mujoco.MjSpec:
  """A static table. Its body origin is the centre of the top surface, so the
  entity's ``init_state.pos`` z is exactly TABLE_HEIGHT.

  Dark walnut rather than the obvious mid-brown: the box is cardboard, and a
  brown box on a brown table is hard to pick out both for a person watching a
  rollout and for any future camera-based variant of this task. The contrast is
  carried by value, not hue.
  """
  spec = mujoco.MjSpec()
  _add_material(
    spec,
    "walnut",
    rgb1=(0.29, 0.20, 0.14),
    rgb2=(0.25, 0.17, 0.11),
    markrgb=(0.20, 0.13, 0.08),
    # Grain running along the table's length.
    texrepeat=(8.0, 2.0),
    texuniform=False,
    specular=0.25,
    shininess=0.3,
  )
  body = spec.worldbody.add_body(name="table")
  body.add_geom(
    name="table_top",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.0, 0.0, -TABLE_SIZE[2] / 2),
    size=(TABLE_SIZE[0] / 2, TABLE_SIZE[1] / 2, TABLE_SIZE[2] / 2),
    material="walnut",
    condim=3,
  )
  # Visual-only legs, so the top does not appear to float. Inset from the
  # corners so they read as legs rather than as walls.
  inset = 0.06
  for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
    body.add_geom(
      name=f"table_leg{i}_visual",
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      fromto=[
        sx * (TABLE_SIZE[0] / 2 - inset),
        sy * (TABLE_SIZE[1] / 2 - inset),
        -TABLE_SIZE[2],
        sx * (TABLE_SIZE[0] / 2 - inset),
        sy * (TABLE_SIZE[1] / 2 - inset),
        -TABLE_HEIGHT,
      ],
      size=[0.022, 0.0, 0.0],
      rgba=(0.20, 0.14, 0.10, 1.0),
      group=2,
      contype=0,
      conaffinity=0,
      density=0.0,
    )
  return spec


##
# Environment config.
##


def dash_lift_box_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Dash upper-body bimanual pick-and-place configuration."""
  cfg = make_lift_cube_env_cfg()

  robot_cfg = get_dash_upper_body_robot_cfg()
  cfg.scene.entities = {
    "robot": robot_cfg,
    "box": EntityCfg(spec_fn=get_box_spec),
    "table": EntityCfg(
      spec_fn=get_table_spec,
      init_state=EntityCfg.InitialStateCfg(pos=(TABLE_CENTER_X, 0.0, TABLE_HEIGHT)),
    ),
  }

  # Box, table and both hands; the box resting on the table alone accounts for
  # four corner contacts. The stock 55 is sized for one arm and a small cube.
  cfg.sim.nconmax = 100

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = DASH_UPPER_BODY_ACTION_SCALE

  # preserve_order so site_ids follow DASH_HAND_SITES (right, then left) rather
  # than the model's declaration order. The observation layout depends on it.
  hands_cfg = SceneEntityCfg("robot", site_names=DASH_HAND_SITES, preserve_order=True)

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
    "hands_to_box": ObservationTermCfg(
      func=mdp.hands_to_object_distance,
      params={"object_name": "box", "asset_cfg": hands_cfg},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "hand_separation": ObservationTermCfg(
      func=mdp.hand_separation,
      params={"asset_cfg": hands_cfg},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "box_to_goal": ObservationTermCfg(
      func=manipulation_mdp.object_to_goal_distance,
      params={"object_name": "box", "command_name": "lift_height"},
      noise=Unoise(n_min=-0.01, n_max=0.01),
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

  # The box is placed by the command term, not by a reset event: the pick and
  # the place have to be sampled together as opposite ends of one traverse, and
  # the command term is the only place that sees both.
  cfg.commands = {
    "lift_height": mdp.TraverseLiftingCommandCfg(
      entity_name="box",
      resampling_time_range=(8.0, 12.0),
      debug_vis=True,
      success_threshold=0.06,
      near_x=PICK_X,
      far_x=PLACE_X,
      lateral_y=LATERAL_Y,
      surface_z=BOX_REST_Z,
      # Not the full circle: the box is oblong in plan and the pads are flat,
      # so a large yaw presents a corner to them instead of a face, and past
      # ~0.4 rad the far face rotates out of the hands' reach entirely.
      # +-0.3 rad still forces the policy to cope with orientation rather than
      # memorizing one approach.
      yaw_range=(-0.3, 0.3),
    )
  }

  ##
  # Sensors.
  ##

  # The stock task terminates on end-effector/floor contact. Here the hands are
  # 0.6 m above the floor and can never touch it, while they are *supposed* to
  # touch the table, so that sensor and its termination are both dropped.
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "ee_ground_collision"
  )
  hand_box_contact_cfg = ContactSensorCfg(
    name="hand_box_contact",
    primary=ContactMatch(mode="geom", pattern=DASH_HAND_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="box", entity="box"),
    fields=("found", "force"),
    # netforce sums every contact point on a pad into one wrench, which is what
    # the grasp threshold should see: a pad flat on a box face makes four
    # corner contacts that individually carry a quarter of the squeeze.
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (hand_box_contact_cfg,)

  ##
  # Events.
  ##

  # Positions the (fixed-base, hence mocap-wrapped) robot and table at their
  # per-environment origins. Without these they all stack at the world origin.
  cfg.events["reset_base"].params["asset_cfg"] = SceneEntityCfg("robot")
  cfg.events["reset_table"] = EventTermCfg(
    func=velocity_mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {},
      "velocity_range": {},
      "asset_cfg": SceneEntityCfg("table"),
    },
  )

  # A little joint scatter at reset. The stock task starts every episode from
  # exactly the same arm pose, which is fine for a 6-DOF arm with a gripper but
  # here leaves the two hands in an identical relationship every time. Capped at
  # 0.03 rad: the ready pose only clears the widest box spawn by 3 cm, and
  # scatter closes the hands by roughly 0.02 m at 0.03 rad, so anything larger
  # starts some episodes with a hand already inside the box.
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.03, 0.03)

  # Retarget the friction randomization from YAM's fingertips to the forearm tips,
  # and rename to match. The slide range is narrowed from the stock (0.3, 1.5):
  # this grip is friction-only -- there are no fingers to wrap around the box --
  # so at mu = 0.3 the box is unholdable at any squeeze the arms can produce and
  # those rollouts teach nothing.
  for axis in ("slide", "spin", "roll"):
    term = cfg.events.pop(f"fingertip_friction_{axis}")
    term.params["asset_cfg"] = SceneEntityCfg("robot", geom_names=DASH_HAND_GEOMS)
    cfg.events[f"hand_friction_{axis}"] = term
  cfg.events["hand_friction_slide"].params["ranges"] = (0.8, 1.5)

  # Mass-density scatter on the box, roughly 0.6x to 1.4x (mass scales as
  # e^(2*alpha)). pseudo_inertia rather than dr.body_mass because it scales the
  # inertia tensor along with the mass; mass alone would leave the box rotating
  # like its original weight, which matters here -- the grip is friction-only,
  # so it is the box twisting out of the pads that loses it, not the box
  # sinking.
  cfg.events["box_inertia"] = EventTermCfg(
    func=dr.pseudo_inertia,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("box", body_names=("box",)),
      "alpha_range": (-0.25, 0.17),
    },
  )

  ##
  # Rewards.
  ##

  cfg.rewards["lift"] = RewardTermCfg(
    func=mdp.bimanual_staged_position_reward,
    weight=1.0,
    params={
      "command_name": "lift_height",
      "object_name": "box",
      # Tighter than the stock 0.2: the hands start ~0.15 m from the box, and a
      # wide kernel is nearly flat over that distance, giving no gradient to
      # follow inward.
      "reaching_std": 0.15,
      "bringing_std": 0.3,
      "asset_cfg": hands_cfg,
    },
  )
  cfg.rewards["lift_precise"].params["object_name"] = "box"

  # The two terms the single-arm task does not need. Without them the policy
  # settles into pushing the box around the table: the distance kernels above
  # are perfectly happy with a box that slides toward the goal's x/y and never
  # leaves the surface.
  cfg.rewards["grasp"] = RewardTermCfg(
    func=mdp.both_hands_in_contact,
    weight=0.5,
    params={"sensor_name": hand_box_contact_cfg.name, "force_threshold": 1.0},
  )
  cfg.rewards["box_height"] = RewardTermCfg(
    func=mdp.object_height_above,
    weight=4.0,
    params={
      "object_name": "box",
      # Half the box above the table, i.e. the height of a box at rest. Only
      # actual clearance scores.
      "reference_height": BOX_REST_Z,
    },
  )

  ##
  # Terminations.
  ##

  cfg.terminations.pop("ee_ground_collision", None)
  # Safety net against solver divergence. Not a task rule: without it a rare
  # NaN propagates into the observations and rsl_rl aborts the whole run.
  # The thresholds are far above anything the task produces -- a held box moves
  # at well under 1 m/s and the arms peak near 6 rad/s under random actions --
  # so this fires on a blow-up and nothing else.
  cfg.terminations["unstable"] = TerminationTermCfg(
    func=mdp.unstable_state,
    params={
      "object_name": "box",
      "max_object_speed": 25.0,
      "max_joint_speed": 100.0,
    },
  )

  cfg.terminations["box_dropped"] = TerminationTermCfg(
    func=mdp.object_dropped,
    params={
      "object_name": "box",
      # Below the table top but above the floor: once the box is over the edge
      # it is gone, since the arms cannot reach down there.
      "minimum_height": TABLE_HEIGHT - 0.15,
    },
  )

  cfg.viewer.body_name = "torso"
  cfg.viewer.distance = 1.6
  cfg.viewer.elevation = -20.0

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    # Resample more often so a play session shows many attempts.
    cfg.commands["lift_height"].resampling_time_range = (5.0, 5.0)
    # Keep the box in play: a dropped box would otherwise end the episode
    # immediately and the viewer would show nothing but resets.
    cfg.terminations.pop("box_dropped", None)

  return cfg
