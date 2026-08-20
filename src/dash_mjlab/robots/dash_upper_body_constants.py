"""Dash upper-body constants.

8 DOF: 4 per arm (shoulder pitch/roll/yaw, elbow pitch). No legs and no hip
joints -- see ``xmls/dash_upper_body.xml``. The torso is welded to the world at
``TORSO_MOUNT_HEIGHT``, so this is a fixed-base entity intended for tabletop
manipulation, not locomotion. Total mass ~19.1 kg.

There are no hands: an object is held by squeezing it between the rounded ends
of the two forearms, so those capsules carry the grip friction (HAND_COLLISION).
"""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from dash_mjlab import DASH_MJLAB_SRC_PATH

##
# MJCF and assets.
##

DASH_UPPER_BODY_XML: Path = (
  DASH_MJLAB_SRC_PATH / "robots" / "xmls" / "dash_upper_body.xml"
)
assert DASH_UPPER_BODY_XML.exists()

# Height of the torso body origin above the floor. Set to the same value the
# standing robot's torso sits at (see KNEES_BENT_KEYFRAME in dash_constants), so
# the arms sweep the same workspace here as they would on the real robot
# standing at a table. Everything downstream -- table height, box spawn, lift
# targets -- is derived from this, so changing it moves the whole task together.
TORSO_MOUNT_HEIGHT = 0.688

# Radius of the visual pedestal standing in for the missing legs. Purely
# cosmetic: it is a visual-class geom, so it never collides.
_PEDESTAL_RADIUS = 0.06


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(DASH_UPPER_BODY_XML))

  # Draw a column from the torso down to the floor. Built here rather than in
  # the MJCF so its length is tied to TORSO_MOUNT_HEIGHT and the two cannot
  # drift apart.
  torso = spec.body("torso")
  torso.add_geom(
    name="pedestal_visual",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    fromto=[0.0, 0.0, 0.0, 0.0, 0.0, -TORSO_MOUNT_HEIGHT],
    size=[_PEDESTAL_RADIUS, 0.0, 0.0],
    rgba=[0.30, 0.30, 0.32, 1.0],
    # Spelled out rather than inherited from the `visual` class, whose geom
    # type is `mesh`. Group 2, no contacts, no mass.
    group=2,
    contype=0,
    conaffinity=0,
    density=0.0,
  )

  return spec


##
# Actuator config.
##

# Same estimate as the full robot's arms (see the warning in dash_constants:
# these are scaled from mass and typical humanoid ratios, not motor specs).
EFFORT_LIMIT_ARM = 30.0
ARMATURE = 0.01

# Stiffer than DASH_ARM_ACTUATOR_CFG's 40. On the walking robot the arms only
# swing, and a soft arm is desirable there -- it keeps arm contacts from
# fighting the gait. Here the arms carry the whole task and there is no
# gravity-compensating base motion to hide sag: one arm is ~4.1 kg
# with its CoM ~0.25 m out, so holding it horizontal needs ~10.5 Nm, which at
# stiffness 40 is 0.26 rad of droop -- larger than the action scale below, i.e.
# the policy would spend all its authority fighting gravity. At 200 the droop is
# ~0.05 rad. Damping is scaled to keep roughly the same damping ratio.
#
# Shoulder yaw and elbow carry less of the arm and are left softer, which also
# makes them more compliant against the box during a squeeze.
DASH_SHOULDER_PITCH_ROLL_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_pitch", ".*_shoulder_roll"),
  stiffness=200.0,
  damping=6.0,
  effort_limit=EFFORT_LIMIT_ARM,
  armature=ARMATURE,
)
DASH_SHOULDER_YAW_ELBOW_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_yaw", ".*_elbow_pitch"),
  stiffness=100.0,
  damping=3.0,
  effort_limit=EFFORT_LIMIT_ARM,
  armature=ARMATURE,
)

##
# Initial state.
##

# Arms reaching slightly forward with the elbows broken, so the hands start
# straddling the box's spawn region and already facing each other. Starting from
# a dead-hang instead makes the first thousand iterations about un-hanging the
# arms rather than about the box.
#
# The roll is what sets the opening between the hands, and it is bounded from
# both sides: adducting less leaves a gap the policy has to close from further
# away, adducting more starts the hands inside the box. At 0.05 rad the hand
# sites sit at y = +-0.166, against a box face that reaches +-0.14 at the
# extreme of the pick range -- 2.6 cm of clearance, which the +-0.03 rad of
# reset scatter (see reset_robot_joints in the task config) eats about a third
# of. Do not raise either without re-checking the other.
#
# This is half the roll the flat-pad version used: the grip surface moved from
# a pad on the end of the forearm to the forearm's own cap, which sits 2.2 cm
# further outboard, so the same roll now starts the hands 2 cm closer together.
ARMS_READY_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, TORSO_MOUNT_HEIGHT),
  joint_pos={
    ".*_shoulder_pitch": -0.3,
    "r_shoulder_roll": 0.0,
    "l_shoulder_roll": 0.0,
    ".*_shoulder_yaw": 0.0,
    ".*_elbow_pitch": -0.4,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_hand_regex = r"^[lr]_lower_arm_collision$"

# The forearms are the hands. Their rounded end caps are the only geoms that can
# hold anything, so they carry the grip parameters and everything else stays a
# frictionless condim-1 collider that exists only to stop the arms passing
# through the torso and the table.
#
# Note this inverts dash_constants' treatment of the same capsules, where the
# arms are frictionless so grazing self-contacts do not fight the gait. Do not
# copy these values back to the walking robot.
#   - condim 4 adds torsional friction. It matters far more here than it would
#     for a flat pad: two spheres pressing on opposite faces make point
#     contacts, and a point contact resists the box spinning between them only
#     through the torsional term.
#   - priority 2 makes the capsule's coefficients win outright over the box's
#     instead of being averaged with them, so the grip does not silently change
#     when the box's friction is randomized.
# 1.2 is plausible for a rubber-sleeved forearm on cardboard but is not
# measured. It is only the nominal value in any case; the manipulation task
# randomizes it per-run over 0.8-1.5 via a `dr.geom_friction` startup event.
HAND_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_hand_regex: 4, ".*_collision": 1},
  priority={_hand_regex: 2},
  friction={_hand_regex: (1.2, 0.02, 0.001)},
)

##
# Final config.
##

DASH_UPPER_BODY_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    DASH_SHOULDER_PITCH_ROLL_ACTUATOR_CFG,
    DASH_SHOULDER_YAW_ELBOW_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_dash_upper_body_robot_cfg() -> EntityCfg:
  """Get a fresh Dash upper-body robot config instance.

  Returns a new EntityCfg each time so callers that mutate it don't affect
  other tasks sharing the config.
  """
  return EntityCfg(
    init_state=ARMS_READY_KEYFRAME,
    collisions=(HAND_COLLISION,),
    spec_fn=get_spec,
    articulation=DASH_UPPER_BODY_ARTICULATION,
  )


# Action scale in rad of joint target per unit of policy output. Smaller than
# the locomotion 0.25: the arm joint ranges are narrow (shoulder pitch spans
# 1.7 rad, shoulder roll 0.9) and a placement task wants fine positioning near
# the box more than it wants large reaching steps.
DASH_UPPER_BODY_ACTION_SCALE: dict[str, float] = {".*": 0.15}

# Sites on the inner surface of each forearm's rounded tip. These are the grasp
# frames the manipulation rewards and observations measure from.
DASH_HAND_SITES: tuple[str, str] = ("r_hand", "l_hand")

# The gripping colliders, for friction randomization and grasp contact sensing.
DASH_HAND_GEOMS: tuple[str, str] = (
  "r_lower_arm_collision",
  "l_lower_arm_collision",
)


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_dash_upper_body_robot_cfg())
  viewer.launch(robot.spec.compile())
