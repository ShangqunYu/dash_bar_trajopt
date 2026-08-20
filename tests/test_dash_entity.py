"""Checks on the Dash entity that catch MJCF/config drift."""

import mujoco
import pytest
from mjlab.entity.entity import Entity

from dash_mjlab.robots import get_dash_robot_cfg

EXPECTED_JOINTS = 18
EXPECTED_MASS_KG = 34.06


@pytest.fixture(scope="module")
def robot() -> Entity:
  return Entity(get_dash_robot_cfg())


def test_spec_compiles(robot: Entity) -> None:
  model = robot.spec.compile()
  assert model.nu == EXPECTED_JOINTS, "every joint should get exactly one actuator"


def test_floating_base(robot: Entity) -> None:
  model = robot.spec.compile()
  assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
  assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 1) == "torso"


def test_total_mass(robot: Entity) -> None:
  model = robot.spec.compile()
  assert model.body_subtreemass[1] == pytest.approx(EXPECTED_MASS_KG, abs=0.1)


def test_actuators_cover_all_joints(robot: Entity) -> None:
  """A regex that matches nothing is the usual silent failure here."""
  model = robot.spec.compile()
  actuated = set()
  for i in range(model.nu):
    jid = model.actuator_trnid[i, 0]
    actuated.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid))
  hinges = {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    for i in range(model.njnt)
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
  }
  assert actuated == hinges


def test_feet_have_sites_and_colliders(robot: Entity) -> None:
  """The velocity task's rewards and sensors address these by name."""
  model = robot.spec.compile()
  for side in ("l", "r"):
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot") >= 0
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
    assert gid >= 0
    assert model.geom_condim[gid] == 3, "feet need friction cone contacts"


def test_spawn_pose_clears_ground(robot: Entity) -> None:
  """Init height should put the soles just above z=0, not intersecting it."""
  model = robot.spec.compile()
  data = mujoco.MjData(model)
  cfg = get_dash_robot_cfg()
  data.qpos[2] = cfg.init_state.pos[2]
  assert cfg.init_state.joint_pos is not None
  for pattern, value in cfg.init_state.joint_pos.items():
    if pattern == ".*":
      continue
    for side in ("l", "r"):
      name = pattern.replace(".*", side)
      jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
      if jid >= 0:
        data.qpos[model.jnt_qposadr[jid]] = value
  mujoco.mj_forward(model, data)

  sole_z = min(
    data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_foot")][2]
    for s in ("l", "r")
  )
  assert 0.0 < sole_z < 0.05, f"soles at z={sole_z:.4f}"
