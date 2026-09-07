"""Checks on the Dash entity that catch MJCF/config drift.

Every check runs against both models: `dash.xml` and the `dash_v2.xml` imported
from the designer's URDF by scripts/import_dash_urdf.py. The whole point of the
v2 import is that it presents the same interface -- same joint, body, geom and
site names -- so the task configs address either one unchanged. Running one
suite over both is what keeps that true.
"""

import mujoco
import pytest
from mjlab.entity.entity import Entity

from dash_mjlab.robots import get_dash_robot_cfg

EXPECTED_JOINTS = 18
EXPECTED_MASS_KG = {False: 34.06, True: 40.00}


@pytest.fixture(scope="module", params=[False, True], ids=["v1", "v2"])
def v2(request: pytest.FixtureRequest) -> bool:
  return request.param


@pytest.fixture(scope="module")
def robot(v2: bool) -> Entity:
  return Entity(get_dash_robot_cfg(v2=v2))


def test_spec_compiles(robot: Entity) -> None:
  model = robot.spec.compile()
  assert model.nu == EXPECTED_JOINTS, "every joint should get exactly one actuator"


def test_floating_base(robot: Entity) -> None:
  model = robot.spec.compile()
  assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
  assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 1) == "torso"


def test_total_mass(robot: Entity, v2: bool) -> None:
  model = robot.spec.compile()
  assert model.body_subtreemass[1] == pytest.approx(EXPECTED_MASS_KG[v2], abs=0.1)


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


def _nominal_data(v2: bool) -> tuple[mujoco.MjModel, mujoco.MjData]:
  """Model and data posed at the configured starting stance."""
  cfg = get_dash_robot_cfg(v2=v2)
  model = Entity(cfg).spec.compile()
  data = mujoco.MjData(model)
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
  return model, data


def test_spawn_pose_clears_ground(v2: bool) -> None:
  """Init height should put the soles just above z=0, not intersecting it."""
  model, data = _nominal_data(v2)
  sole_z = min(
    data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_foot")][2]
    for s in ("l", "r")
  )
  assert 0.0 < sole_z < 0.05, f"soles at z={sole_z:.4f}"


def test_no_self_collision_at_nominal_pose(v2: bool) -> None:
  """The starting stance must not begin already in self-contact.

  `self_collisions` carries weight -1.0, so a collider pair that overlaps at the
  spawn pose bills the policy from step one for something it cannot undo. This
  is easy to reintroduce when refitting collision primitives, since the arms
  hang close enough to the hips that round capsules overlap where the real links
  clear -- see the contact excludes in dash_v2.xml.
  """
  model, data = _nominal_data(v2)
  contacts = [
    (
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom1),
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom2),
    )
    for i in range(data.ncon)
  ]
  assert contacts == [], f"self-contacts at the spawn pose: {contacts}"


def test_v1_and_v2_expose_the_same_names() -> None:
  """The task configs match on these by regex, so both models must carry them."""
  models = {
    v2: Entity(get_dash_robot_cfg(v2=v2)).spec.compile() for v2 in (False, True)
  }

  def names(model: mujoco.MjModel, objtype: mujoco.mjtObj, count: int) -> set[str]:
    found = {mujoco.mj_id2name(model, objtype, i) for i in range(count)}
    return {n for n in found if n}

  for objtype, count_attr in (
    (mujoco.mjtObj.mjOBJ_JOINT, "njnt"),
    (mujoco.mjtObj.mjOBJ_BODY, "nbody"),
    (mujoco.mjtObj.mjOBJ_SITE, "nsite"),
    (mujoco.mjtObj.mjOBJ_SENSOR, "nsensor"),
  ):
    v1_names = names(models[False], objtype, getattr(models[False], count_attr))
    v2_names = names(models[True], objtype, getattr(models[True], count_attr))
    assert v1_names == v2_names, (
      f"{objtype} differs: only in v1={sorted(v1_names - v2_names)}, "
      f"only in v2={sorted(v2_names - v1_names)}"
    )

  # Geoms are compared on colliders alone: the visual geoms are named after the
  # meshes, which the two exports name differently, and nothing matches on them.
  colliders = {
    v2: {
      n
      for n in names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)
      if n.endswith("_collision")
    }
    for v2, model in models.items()
  }
  assert colliders[False] == colliders[True], (
    f"collision geoms differ: only in v1={sorted(colliders[False] - colliders[True])}, "
    f"only in v2={sorted(colliders[True] - colliders[False])}"
  )
