"""Dash humanoid constants.

18 DOF: 5 per leg (hip yaw/roll/pitch, knee pitch, ankle pitch) and 4 per arm
(shoulder pitch/roll/yaw, elbow pitch). Total mass ~34.1 kg.
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

DASH_XML: Path = DASH_MJLAB_SRC_PATH / "robots" / "xmls" / "dash.xml"
assert DASH_XML.exists()


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(DASH_XML))

  # Every joint in the MJCF carries `actuatorfrcrange="-10 10"`, inherited from
  # the URDF's placeholder 10 Nm. MuJoCo applies that clamp to qfrc_actuator
  # *independently* of the actuator's own forcerange, so the tighter of the two
  # wins and the effort_limit values below become dead code. At 10 Nm the robot
  # cannot stand: holding the default pose needs ~42 Nm at hip_pitch, so the hip
  # saturates and sags 0.58 rad, and every episode begins mid-collapse.
  #
  # Clearing the joint-level limit makes effort_limit the single source of truth.
  # Re-derive both from real motor specs before deploying to hardware.
  for joint in spec.joints:
    joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_FALSE

  return spec


##
# Actuator config.
##

# WARNING: the effort limits below are ESTIMATES. Neither
# the URDF (placeholder 10 Nm on every joint) nor the DashMotorControl firmware
# (KT and gear ratio are runtime-configurable registers) pins them down. They
# are scaled from the robot's 34 kg mass and typical humanoid ratios. Replace
# with real values before deploying -- they bound the
# torque the policy can rely on, so a policy trained against optimistic limits
# will not reproduce on hardware.
EFFORT_LIMIT_HIP_YAW_ROLL = 60.0
EFFORT_LIMIT_HIP_KNEE_PITCH = 120.0
EFFORT_LIMIT_ANKLE = 40.0
EFFORT_LIMIT_ARM = 30.0

# Reflected rotor inertia, also carried over from IsaacLab config.
ARMATURE = 0.01

# Sanity check for any future change: stiffness must exceed
# (gravity torque at that joint) / (acceptable tracking error). Measured by
# settling the default pose under contact with zero action (see the settle test
# in scripts/), hip_pitch carries 42 Nm and everything else is under 5 Nm --
# the earlier ~24 Nm estimate was taken while the joints were still clamped at
# 10 Nm, so it read the clamp rather than the load.
#
# At stiffness 200 that 42 Nm leaves the hip sagging 0.18 rad while merely
# standing. With an action scale of 0.25 rad, gravity compensation alone would
# eat 72% of the policy's authority before it does anything useful. 350 brings
# the sag to 0.12 rad; past ~400 it flattens out and is not worth the stiffer
# contact dynamics. Damping is scaled with sqrt(stiffness) to hold the same
# damping ratio.
DASH_HIP_YAW_ROLL_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw", ".*_hip_roll"),
  stiffness=150.0,
  damping=5.0,
  effort_limit=EFFORT_LIMIT_HIP_YAW_ROLL,
  armature=ARMATURE,
)
DASH_HIP_KNEE_PITCH_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch", ".*_knee_pitch"),
  stiffness=350.0,
  damping=8.0,
  effort_limit=EFFORT_LIMIT_HIP_KNEE_PITCH,
  armature=ARMATURE,
)
# Dash has no ankle roll, so the ankles are the only joints that can shift the
# centre of pressure fore/aft, and the foot box is only 0.2 m long. Holding the
# CoP at the toe needs ~33 Nm; at stiffness 40 that is 0.83 rad of deflection,
# i.e. the ankle folds before it can arrest a forward lean. 80 halves it.
DASH_ANKLE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch",),
  stiffness=80.0,
  damping=2.0,
  effort_limit=EFFORT_LIMIT_ANKLE,
  armature=ARMATURE,
)
DASH_ARM_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch",
    ".*_shoulder_roll",
    ".*_shoulder_yaw",
    ".*_elbow_pitch",
  ),
  stiffness=40.0,
  damping=1.0,
  effort_limit=EFFORT_LIMIT_ARM,
  armature=ARMATURE,
)

# Starting stance from IsaacLab config
KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.688), # Puts foot at z=0
  joint_pos={
    ".*_hip_pitch": -0.349,
    ".*_knee_pitch": 0.698,
    ".*_ankle_pitch": -0.349,
    ".*_elbow_pitch": -0.8, # flexes elbow forward/up
    ".*_shoulder_pitch": 0.4,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = r"^[lr]_foot_collision$"

# Everything collides. Feet get condim=3 and friction priority; every other
# collider is frictionless (condim=1) so grazing self-contacts don't fight the
# gait.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_foot_regex: 3, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.6,)},
)

# Feet only, no self collision. Cheaper; useful for flat-ground training.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

DASH_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    DASH_HIP_YAW_ROLL_ACTUATOR_CFG,
    DASH_HIP_KNEE_PITCH_ACTUATOR_CFG,
    DASH_ANKLE_ACTUATOR_CFG,
    DASH_ARM_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_dash_robot_cfg() -> EntityCfg:
  """Get a fresh Dash robot config instance.

  Returns a new EntityCfg each time so callers that mutate it don't affect
  other tasks sharing the config.
  """
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=DASH_ARTICULATION,
  )


# Action scale in rad of joint target per unit of policy output.
#
# mjlab's built-in robots derive this as 0.25 * effort_limit / stiffness, which
# only means something when the effort limit is a real motor spec. Dash's are
# estimates, so the scale is set directly instead: 0.25 rad is the
# usual starting point for humanoid locomotion. Revisit once real torque limits
# are calculated.
DASH_ACTION_SCALE: dict[str, float] = {".*": 0.25}


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_dash_robot_cfg())
  viewer.launch(robot.spec.compile())
