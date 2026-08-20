"""Checks on the Dash upper-body entity and the workspace the task assumes.

The manipulation task's box spawn and lift targets are numbers picked off a
sweep of the compiled arms (see the WORKSPACE_* constants). Nothing at runtime
re-derives them, so an edit to the MJCF -- a joint range, a link offset, the pad
placement -- would leave the task silently commanding poses the arms cannot
reach. ``test_task_workspace_is_reachable`` is the guard for that.
"""

import mujoco
import numpy as np
import pytest
from mjlab.entity.entity import Entity

from dash_mjlab.robots import (
  DASH_HAND_GEOMS,
  DASH_HAND_SITES,
  TORSO_MOUNT_HEIGHT,
  get_dash_upper_body_robot_cfg,
)
from dash_mjlab.tasks.manipulation.env_cfgs import (
  BOX_HALF_EXTENTS,
  BOX_REST_Z,
  LATERAL_Y,
  PICK_X,
  PLACE_X,
  TABLE_CENTER_X,
  TABLE_HEIGHT,
  get_box_spec,
  get_table_spec,
)

# Height the box is carried at between the two ends. Not a task constant -- the
# policy picks its own transit height -- but the arms have to be able to hold
# the box somewhere clear of the surface or there is no way across that is not
# a drag. 6 cm puts the box's underside a full 6 cm above the table, which is
# ample, and it is about as high as the near corner can be carried: the strip
# narrows as it rises, and by 8 cm the near corner at full lateral offset has
# dropped below the grip threshold. Carrying higher is possible over the rest
# of the strip, just not at that one corner.
TRANSIT_Z = BOX_REST_Z + 0.06

EXPECTED_JOINTS = 8
EXPECTED_MASS_KG = 19.08

# Leg link and joint name fragments. Anything matching means the strip was
# incomplete.
LEG_FRAGMENTS = ("hip", "leg", "knee", "ankle", "foot")


@pytest.fixture(scope="module")
def robot() -> Entity:
  return Entity(get_dash_upper_body_robot_cfg())


@pytest.fixture(scope="module")
def model(robot: Entity) -> mujoco.MjModel:
  return robot.spec.compile()


def test_spec_compiles(model: mujoco.MjModel) -> None:
  assert model.nu == EXPECTED_JOINTS, "every joint should get exactly one actuator"
  assert model.nq == EXPECTED_JOINTS, "fixed base: no freejoint coordinates"


def test_fixed_base(robot: Entity) -> None:
  """No freejoint, so mjlab wraps it in a mocap body it can place per-env."""
  assert robot.is_fixed_base
  assert robot.is_mocap, (
    "a fixed-base entity that is not mocap cannot be moved off the world origin, "
    "so every environment would stack the robot in the same place"
  )


def test_total_mass(model: mujoco.MjModel) -> None:
  assert model.body_subtreemass[1] == pytest.approx(EXPECTED_MASS_KG, abs=0.1)


def test_no_legs(model: mujoco.MjModel) -> None:
  for obj_type, count in (
    (mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
    (mujoco.mjtObj.mjOBJ_BODY, model.nbody),
    (mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
    (mujoco.mjtObj.mjOBJ_MESH, model.nmesh),
  ):
    for i in range(count):
      name = mujoco.mj_id2name(model, obj_type, i) or ""
      assert not any(f in name.lower() for f in LEG_FRAGMENTS), (
        f"leg element {name!r} survived the strip"
      )


def test_actuators_cover_all_joints(model: mujoco.MjModel) -> None:
  """A regex that matches nothing is the usual silent failure here."""
  actuated = {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i, 0])
    for i in range(model.nu)
  }
  hinges = {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    for i in range(model.njnt)
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
  }
  assert actuated == hinges


def test_hands_have_sites_and_grip_contacts(model: mujoco.MjModel) -> None:
  """The task's grasp reward and contact sensor address these by name."""
  for site in DASH_HAND_SITES:
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site) >= 0
  for geom in DASH_HAND_GEOMS:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
    assert gid >= 0
    assert model.geom_condim[gid] == 4, (
      "a two-point ball grip resists the box spinning only through torsional "
      "friction, so condim 4 is not optional here"
    )
    assert model.geom_priority[gid] > 0, "hand friction must win over the object's"


def test_hands_face_each_other(model: mujoco.MjModel) -> None:
  """Each grasp site is on the inboard side of its forearm's rounded end.

  A squeeze grip only exists if the two contact points oppose each other across
  the midline. A sign flip on one side's offset would put its site on the
  outboard face -- the arms would then be rewarded for reaching to a point
  behind their own hand, and nothing else in the model or the config would
  complain.
  """
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)

  for site_name, geom_name, inboard in (
    ("r_hand", "r_lower_arm_collision", +1.0),
    ("l_hand", "l_lower_arm_collision", -1.0),
  ):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    offset = data.site_xpos[site_id][1] - data.geom_xpos[geom_id][1]
    assert inboard * offset > 0.0, (
      f"{site_name} sits outboard of its forearm (offset {offset:+.4f})"
    )
    # And on the surface, not buried in the capsule or floating off it.
    radius = model.geom_size[geom_id][0]
    assert abs(abs(offset) - radius) < 0.02, (
      f"{site_name} is {abs(offset):.3f} m off the forearm axis, "
      f"but the capsule radius is {radius:.3f}"
    )


def test_ready_pose_clears_the_box(model: mujoco.MjModel) -> None:
  """The hands must start outside the widest box spawn, or resets punch the box.

  The margin also has to survive the reset joint scatter, which is why it is
  checked against a clearance rather than mere non-overlap.
  """
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)

  r_y = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r_hand")][1]
  l_y = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "l_hand")][1]
  # Furthest a box face can be from the midline: half-width plus spawn offset.
  box_face = BOX_HALF_EXTENTS[1] + LATERAL_Y[1]
  assert -r_y - box_face > 0.02, f"right hand at y={r_y:.3f}, box face at {-box_face}"
  assert l_y - box_face > 0.02, f"left hand at y={l_y:.3f}, box face at {box_face}"


def _grip_scene() -> mujoco.MjModel:
  """The robot, the task's table and the task's box, in one model.

  Built from the task's own spec functions so the test cannot drift from the
  configuration it is meant to guard.
  """
  spec = Entity(get_dash_upper_body_robot_cfg()).spec
  spec.body("mocap_base").pos = np.array([0.0, 0.0, TORSO_MOUNT_HEIGHT])
  spec.body("mocap_base").mocap = False
  table_frame = spec.worldbody.add_frame(pos=[TABLE_CENTER_X, 0.0, TABLE_HEIGHT])
  spec.attach(child=get_table_spec(), prefix="table/", frame=table_frame)
  box_frame = spec.worldbody.add_frame(pos=[0.0, 0.0, 0.0])
  spec.attach(child=get_box_spec(), prefix="box/", frame=box_frame)

  model = spec.compile()
  model.opt.timestep = 0.005
  model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  model.opt.impratio = 10
  return model


def _squeeze(model: mujoco.MjModel, x: float, y: float, z: float) -> float:
  """Put the box at (x, y, z), drive the hands hard into it, return grip force.

  Drives the hand targets 4 cm inside each face, which is well past anything the
  arms can achieve, so the result is the strongest squeeze available at that
  point rather than one particular controller's.
  """
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  box_adr = model.joint("box/box_joint").qposadr[0]
  data.qpos[box_adr : box_adr + 7] = [x, y, z, 1, 0, 0, 0]
  mujoco.mj_forward(model, data)

  # Above the surface there is nothing holding the box up, so without this it
  # would fall back to the table before the hands close and the check would
  # silently measure the table-height case instead.
  box_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box/box")
  gravcomp = float(model.body_gravcomp[box_body])
  if z > BOX_REST_Z + 1e-6:
    model.body_gravcomp[box_body] = 1.0

  site_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in DASH_HAND_SITES
  ]
  lower, upper = model.jnt_range[:8, 0], model.jnt_range[:8, 1]
  offset = BOX_HALF_EXTENTS[1] - 0.04
  targets = (np.array([x, y - offset, z]), np.array([x, y + offset, z]))

  # Damped least squares onto those targets.
  scratch = mujoco.MjData(model)
  scratch.qpos[:] = data.qpos
  q = data.qpos[:8].copy()
  jacp = np.zeros((3, model.nv))
  for _ in range(300):
    scratch.qpos[:8] = q
    mujoco.mj_kinematics(model, scratch)
    mujoco.mj_comPos(model, scratch)
    errors, jacobians = [], []
    for site_id, target in zip(site_ids, targets, strict=True):
      mujoco.mj_jacSite(model, scratch, jacp, None, site_id)
      errors.append(target - scratch.site_xpos[site_id])
      jacobians.append(jacp[:, :8].copy())
    jac = np.vstack(jacobians)
    step = jac.T @ np.linalg.solve(
      jac @ jac.T + 1e-4 * np.eye(6), np.concatenate(errors)
    )
    q = np.clip(q + 0.5 * step, lower, upper)

  # mjlab groups actuators by actuator-cfg declaration order, not joint order.
  ctrl_of_joint = {int(model.actuator_trnid[i, 0]): i for i in range(model.nu)}
  data.ctrl[[ctrl_of_joint[j] for j in range(8)]] = q
  for _ in range(500):
    mujoco.mj_step(model, data)

  model.body_gravcomp[box_body] = gravcomp
  hand_ids = {
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in DASH_HAND_GEOMS
  }
  box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box/box_geom")
  total = 0.0
  for i in range(data.ncon):
    contact = data.contact[i]
    if box_id not in (contact.geom1, contact.geom2):
      continue
    other = contact.geom1 if contact.geom2 == box_id else contact.geom2
    if other in hand_ids:
      force = np.zeros(6)
      mujoco.mj_contactForce(model, data, i, force)
      total += abs(force[0])
  return total


@pytest.fixture(scope="module")
def grip_scene() -> mujoco.MjModel:
  return _grip_scene()


def _traverse_corners() -> list[tuple[float, float, float]]:
  """Every corner the command term can pick or place at, plus transit height."""
  xs = (*PICK_X, *PLACE_X)
  ys = (-LATERAL_Y[1], -LATERAL_Y[0], LATERAL_Y[0], LATERAL_Y[1])
  return [(x, y, z) for x in xs for y in ys for z in (BOX_REST_Z, TRANSIT_Z)]


@pytest.mark.parametrize(
  "corner", _traverse_corners(), ids=lambda c: "x%.2f_y%+.2f_z%.2f" % c
)
def test_grip_forms_across_the_traverse(
  grip_scene: mujoco.MjModel, corner: tuple[float, float, float]
) -> None:
  """A squeeze actually grips the box everywhere the task puts one.

  This simulates the grip rather than checking that IK can place the hands,
  because reach is not the binding constraint -- torque is. IK solves poses near
  the torso exactly, while the servos settle 6-11 cm wider than commanded and
  the pads never touch the box. An earlier position-only version of this test
  passed a pick zone where no grip was possible at all.

  The threshold is deliberately low: 10 N is far below the 30-75 N the strip
  produces and far above the zero seen outside it, so this fails on a design
  error and not on a few newtons of solver drift.
  """
  x, y, z = corner
  force = _squeeze(grip_scene, x, y, z)
  assert force > 10.0, (
    f"no usable grip at ({x:.2f}, {y:+.2f}, {z:.2f}): {force:.1f} N on the pads"
  )


def test_traverse_is_actually_long(model: mujoco.MjModel) -> None:
  """The pick and place ends must not overlap.

  The whole point of the custom command term is that every episode is a real
  traverse. If someone widens PICK_X or PLACE_X until they meet, the task
  quietly degrades back into a short hop and nothing else notices.
  """
  del model
  assert PICK_X[1] < PLACE_X[0], "pick and place ranges overlap"
  # Worst case: pick at the far edge of the near zone and place at the near edge
  # of the far zone, both at the smallest lateral offset.
  shortest = float(np.hypot(PLACE_X[0] - PICK_X[1], 2 * LATERAL_Y[0]))
  assert shortest > 0.06, f"shortest traverse is only {shortest * 100:.1f} cm"


def test_torso_mounted_at_configured_height() -> None:
  cfg = get_dash_upper_body_robot_cfg()
  assert cfg.init_state.pos[2] == TORSO_MOUNT_HEIGHT
