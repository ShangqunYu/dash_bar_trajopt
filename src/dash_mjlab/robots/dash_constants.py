"""Dash humanoid constants.

18 DOF: 5 per leg (hip yaw/roll/pitch, knee pitch, ankle pitch) and 4 per arm
(shoulder pitch/roll/yaw, elbow pitch). Total mass ~34.1 kg.
"""

import functools
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

# Generated from the designer's robotDataPackage URDF by
# scripts/import_dash_urdf.py. Same 18 DOF and the same body, joint, geom and
# site names as dash.xml, so every regex in the task configs matches both, but
# the masses, inertias, link geometry, joint limits and several joint sign
# conventions are the designer's rather than the older export's. See
# KNEES_BENT_KEYFRAME_V2 for the sign difference that reaches this file.
DASH_V2_XML: Path = DASH_MJLAB_SRC_PATH / "robots" / "xmls" / "dash_v2.xml"
assert DASH_V2_XML.exists()


def get_spec(xml: Path = DASH_XML) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(xml))

  # Every joint in the MJCF carries `actuatorfrcrange="-10 10"`, inherited from
  # the URDF's placeholder 10 Nm. MuJoCo applies that clamp to qfrc_actuator
  # *independently* of the actuator's own forcerange, so the tighter of the two
  # wins and the effort_limit values below become dead code. At 10 Nm the robot
  # cannot stand: holding the default pose needs ~42 Nm at hip_pitch, so the hip
  # saturates and sags 0.58 rad, and every episode begins mid-collapse.
  #
  # Clearing the joint-level limit makes effort_limit the single source of truth.
  # Re-derive both from real motor specs before deploying to hardware.
  #
  # dash_v2.xml carries no actuatorfrcrange at all -- the designer's URDF leaves
  # effort="0" on every joint, which is no more usable as a limit than the 10 Nm
  # placeholder -- so this loop is a no-op there and effort_limit already rules.
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

# Gains matched to the Unitree G1's published sim-to-real config
# (unitree_rl_gym deploy/deploy_real/configs/g1.yaml: legs 100/150 stiffness,
# 2/4 damping; arms 20-100 stiffness, 1-2 damping) rather than derived from
# Dash's own load, on the user's call that the earlier per-joint tuning
# below -- sized to hold gravity sag under 0.12-0.18 rad -- ran stiffer than
# the real actuators. Trade-off to watch: hip_pitch measured at 42 Nm holding
# the default stance (see the settle test in scripts/), so at stiffness 80 it
# will sag ~0.5 rad under gravity alone -- the policy has to actively fight
# that rather than get it for free from the PD term, unlike the old 350
# setting. If training stalls on just holding a standing pose, this is the
# first thing to revisit.
DASH_HIP_YAW_ROLL_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw", ".*_hip_roll"),
  stiffness=80.0,
  damping=2.0,
  effort_limit=EFFORT_LIMIT_HIP_YAW_ROLL,
  armature=ARMATURE,
)
DASH_HIP_KNEE_PITCH_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch", ".*_knee_pitch"),
  stiffness=80.0,
  damping=2.0,
  effort_limit=EFFORT_LIMIT_HIP_KNEE_PITCH,
  armature=ARMATURE,
)
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

# Same physical stance on the v2 model. Two things change:
#
#   - 0.6735 rather than 0.688, because the designer's legs put the sole that
#     far below the torso at this pose. Measured, not guessed: fold the
#     keyframe into the model and read off the foot sites.
#   - shoulder_pitch is negated. Six of the eighteen joints have the opposite
#     sign convention in the designer's URDF (l_hip_yaw, l_shoulder_pitch,
#     r_hip_roll, r_shoulder_pitch, r_shoulder_roll, r_shoulder_yaw), and
#     shoulder_pitch is the only one of those with a non-zero default here. The
#     rest of the pose is sign-identical. Both sides flip together, so the
#     stance stays mirror-symmetric; +0.4 would swing both arms backwards.
#     Verified by comparing forearm position against dash.xml at this pose.
KNEES_BENT_KEYFRAME_V2 = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.6735),
  joint_pos={
    ".*_hip_pitch": -0.349,
    ".*_knee_pitch": 0.698,
    ".*_ankle_pitch": -0.349,
    ".*_elbow_pitch": -0.8,
    ".*_shoulder_pitch": -0.4,
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


def get_dash_robot_cfg(v2: bool = False) -> EntityCfg:
  """Get a fresh Dash robot config instance.

  Returns a new EntityCfg each time so callers that mutate it don't affect
  other tasks sharing the config.

  Set `v2` for the model imported from the designer's URDF. The two share every
  name the task configs match on, so the collision and actuator configs above
  apply unchanged; only the model file and the starting stance differ.
  """
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME_V2 if v2 else KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=functools.partial(get_spec, DASH_V2_XML if v2 else DASH_XML),
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
