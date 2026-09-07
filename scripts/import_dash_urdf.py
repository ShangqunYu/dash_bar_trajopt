"""Convert the designer's Dash URDF into the repo's MJCF + decimated assets.

The URDF shipped in `robotDataPackage` is a CAD export, not a sim model. It
needs five things done to it before mjlab can use it, all of them mechanical
and all of them redone here rather than by hand so the import stays
reproducible when the designer sends a new revision:

  1. `base_rot_z` welds `torso` to a `world` link. MuJoCo merges a fixed child
     into its parent, which deletes `torso` as a body and moves its 11.4 kg into
     the static world -- the model loads, simulates, and is silently bolted to
     the origin with 28.6 of its 40 kg. Dropped, and a freejoint put on `torso`.
  2. `<axis xyz="100"/>` on that same joint is one token where three are
     required; MuJoCo refuses the whole file. Moot once the joint is dropped.
  3. Mesh paths use Windows separators (`meshes\\visual\\x.stl`), which are
     literal characters on Linux.
  4. Link and joint names use `left_`/`right_`; every regex in this repo
     (actuator targets, reward terms, contact sensors, collision configs)
     matches `l_`/`r_`. Renamed here rather than loosening ~30 patterns.
  5. The supplied collision meshes are byte-identical copies of the visual
     meshes -- 590k triangles, 91.5k on the torso alone. Replaced with fitted
     primitives (capsules on limbs, boxes on torso and feet), matching the geom
     set the task configs already target by name.

Deliberately NOT changed, since the URDF is authoritative: joint axes and their
sign conventions, joint limits, link masses and inertias, and link geometry.

Usage:
  uv run --with fast-simplification python scripts/import_dash_urdf.py \
      --src ~/Downloads/robotDataPackage
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh

REPO = Path(__file__).parent.parent
DEFAULT_XML = REPO / "src/dash_mjlab/robots/xmls/dash_v2.xml"
DEFAULT_ASSETS = REPO / "src/dash_mjlab/robots/xmls/assets_v2"

# The designer's names on the left, this repo's on the right. Applied to link
# names, joint names and mesh filenames alike.
LINK_RENAME = {
  "torso": "torso",
  "prox_hip": "hip",
  "dist_hip": "dist_hip",
  "upper_leg": "upper_leg",
  "lower_leg": "lower_leg",
  "foot": "foot",
  "prox_shoulder": "prox_shoulder",
  "dist_shoulder": "dist_shoulder",
  "upper_arm": "upper_arm",
  "forearm": "lower_arm",
}
# `knee`/`ankle`/`elbow` gain the `_pitch` suffix the actuator regexes expect.
JOINT_RENAME = {
  "hip_yaw": "hip_yaw",
  "hip_roll": "hip_roll",
  "hip_pitch": "hip_pitch",
  "knee": "knee_pitch",
  "ankle": "ankle_pitch",
  "shoulder_pitch": "shoulder_pitch",
  "shoulder_roll": "shoulder_roll",
  "shoulder_yaw": "shoulder_yaw",
  "elbow": "elbow_pitch",
}

# Which primitive stands in for each link's collision mesh. Links absent here
# get no collider, matching dash.xml: the shoulder links sit inside the torso's
# swept volume and never reach the ground on their own.
CAPSULE_LINKS = (
  "hip",
  "dist_hip",
  "upper_leg",
  "lower_leg",
  "upper_arm",
  "lower_arm",
)
BOX_LINKS = ("torso", "foot")

# Radial percentile used for capsule radius. The max would size every capsule to
# the single most protruding bolt head; 92% tracks the shaft and keeps the
# capsules from swallowing their neighbours.
CAPSULE_RADIUS_PCT = 92.0

# Torso box percentile. Its mesh AABB is 227 mm deep because it includes the hip
# mounts protruding off the back; clipping to the bulk of the shell keeps the
# collider off the upper-leg capsules, which are grandchildren and so are not
# auto-excluded from contact.
TORSO_EXTENT_PCT = 98.0

# Foot sole slab. Only the bottom of the foot mesh is a contact surface -- its
# AABB is 64 mm tall and laterally offset because it includes the ankle bracket.
# Vertices within this height of the lowest point define the sole's footprint.
SOLE_BAND = 0.015
SOLE_HALF_THICKNESS = 0.0125


def rename(name: str) -> str:
  """Map a designer link/joint name onto this repo's convention."""
  for prefix, short in (("left_", "l_"), ("right_", "r_")):
    if name.startswith(prefix):
      stem = name[len(prefix) :]
      table = JOINT_RENAME if stem in JOINT_RENAME else LINK_RENAME
      return short + table.get(stem, stem)
  return LINK_RENAME.get(name, name)


def rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
  """URDF fixed-axis roll-pitch-yaw to a MuJoCo (w, x, y, z) quaternion."""
  r, p, y = rpy
  cr, sr = np.cos(r / 2), np.sin(r / 2)
  cp, sp = np.cos(p / 2), np.sin(p / 2)
  cy, sy = np.cos(y / 2), np.sin(y / 2)
  return np.array(
    [
      cr * cp * cy + sr * sp * sy,
      sr * cp * cy - cr * sp * sy,
      cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy,
    ]
  )


def fmt(values: Iterable[float], places: int = 6) -> str:
  """Format a vector for XML, trimming float noise that hurts readability."""
  out = []
  for v in values:
    v = 0.0 if abs(v) < 10.0**-places else v
    out.append(f"{v:.{places}g}")
  return " ".join(out)


@dataclass
class Link:
  name: str
  mass: float
  com: np.ndarray
  inertia: np.ndarray  # ixx iyy izz ixy ixz iyz, about the COM.
  mesh: str | None


@dataclass
class Joint:
  name: str
  parent: str
  child: str
  pos: np.ndarray
  quat: np.ndarray
  axis: np.ndarray
  lower: float
  upper: float


def parse_urdf(path: Path) -> tuple[dict[str, Link], list[Joint]]:
  root = ET.parse(path).getroot()

  links: dict[str, Link] = {}
  for el in root.findall("link"):
    name = rename(el.get("name", ""))
    inertial = el.find("inertial")
    if inertial is None:
      continue
    mass = float(inertial.find("mass").get("value"))
    if mass <= 0.0:  # The placeholder `world` link.
      continue
    origin = inertial.find("origin")
    com = (
      np.fromstring(origin.get("xyz"), sep=" ") if origin is not None else np.zeros(3)
    )
    i = inertial.find("inertia")
    inertia = np.array(
      [
        float(i.get("ixx")),
        float(i.get("iyy")),
        float(i.get("izz")),
        float(i.get("ixy")),
        float(i.get("ixz")),
        float(i.get("iyz")),
      ]
    )
    visual = el.find("visual")
    mesh = None
    if visual is not None:
      mesh_el = visual.find("geometry/mesh")
      if mesh_el is not None:
        mesh = Path(mesh_el.get("filename").replace("\\", "/")).stem
    links[name] = Link(name, mass, com, inertia, mesh)

  joints: list[Joint] = []
  for el in root.findall("joint"):
    if el.get("type") != "revolute":
      continue  # The only non-revolute joint is the world weld we are dropping.
    origin = el.find("origin")
    limit = el.find("limit")
    joints.append(
      Joint(
        name=rename(el.get("name", "")),
        parent=rename(el.find("parent").get("link")),
        child=rename(el.find("child").get("link")),
        pos=np.fromstring(origin.get("xyz"), sep=" "),
        quat=rpy_to_quat(np.fromstring(origin.get("rpy"), sep=" ")),
        axis=np.fromstring(el.find("axis").get("xyz"), sep=" "),
        lower=float(limit.get("lower")),
        upper=float(limit.get("upper")),
      )
    )
  return links, joints


def fit_capsule(
  mesh: trimesh.Trimesh, pct: float = CAPSULE_RADIUS_PCT
) -> tuple[np.ndarray, np.ndarray, float]:
  """Fit a capsule to a limb mesh, in the mesh's own (link) frame.

  Returns the two segment endpoints and the radius. The long axis comes from a
  PCA of the vertices; the radius is a high percentile of the radial distance
  off that axis, and the endpoints are pulled in by the radius so the capsule's
  overall length matches the mesh's extent rather than overshooting it by a
  hemisphere at each end.
  """
  v = mesh.vertices - mesh.vertices.mean(axis=0)
  axis = np.linalg.svd(v, full_matrices=False)[2][0]

  t = v @ axis
  radial = np.linalg.norm(v - np.outer(t, axis), axis=1)
  radius = float(np.percentile(radial, pct))

  centre = mesh.vertices.mean(axis=0)
  lo, hi = float(t.min()), float(t.max())
  # Inset, but never past the midpoint of a stubby link.
  inset = min(radius, (hi - lo) / 2.0 - 1e-4)
  return centre + axis * (lo + inset), centre + axis * (hi - inset), radius


def fit_box(mesh: trimesh.Trimesh, pct: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
  """Fit an axis-aligned box to a mesh, in the mesh's own (link) frame.

  `pct` trims protruding brackets: at 98 the box spans the 1st to 99th
  percentile of vertices on each axis instead of the full extent.
  """
  tail = (100.0 - pct) / 2.0
  lo = np.percentile(mesh.vertices, tail, axis=0)
  hi = np.percentile(mesh.vertices, 100.0 - tail, axis=0)
  return (lo + hi) / 2.0, (hi - lo) / 2.0


def quat_to_mat(q: np.ndarray) -> np.ndarray:
  """MuJoCo (w, x, y, z) quaternion to a rotation matrix."""
  w, x, y, z = q
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
  )


def mat_to_quat(r: np.ndarray) -> np.ndarray:
  """Rotation matrix to a MuJoCo (w, x, y, z) quaternion."""
  t = np.trace(r)
  if t > 0:
    s = np.sqrt(t + 1.0) * 2
    return np.array(
      [
        0.25 * s,
        (r[2, 1] - r[1, 2]) / s,
        (r[0, 2] - r[2, 0]) / s,
        (r[1, 0] - r[0, 1]) / s,
      ]
    )
  i = int(np.argmax(np.diag(r)))
  j, k = (i + 1) % 3, (i + 2) % 3
  s = np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2
  q = np.empty(4)
  q[0] = (r[k, j] - r[j, k]) / s
  q[1 + i] = 0.25 * s
  q[1 + j] = (r[j, i] + r[i, j]) / s
  q[1 + k] = (r[k, i] + r[i, k]) / s
  return q


def forward_kinematics(joints: list[Joint]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
  """World pose of every link at the zero pose, with the torso frame as origin."""
  joint_to_child = {j.child: j for j in joints}
  poses: dict[str, tuple[np.ndarray, np.ndarray]] = {"torso": (np.eye(3), np.zeros(3))}

  def resolve(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name in poses:
      return poses[name]
    j = joint_to_child[name]
    rp, pp = resolve(j.parent)
    # At q=0 the joint contributes no rotation of its own.
    pose = (rp @ quat_to_mat(j.quat), pp + rp @ j.pos)
    poses[name] = pose
    return pose

  for j in joints:
    resolve(j.child)
  return poses


def fit_sole(
  mesh: trimesh.Trimesh, rot: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Fit a thin contact slab to the underside of a foot, in the link frame.

  Only the sole of the foot is a collider. Which way is "down" cannot be read
  off the link frame: the leg chain is rotated, and the two foot meshes are
  mirrored relative to each other, so the left foot's sole sits at its minimum
  local z while the right foot's sits at its maximum. `rot` is the foot's world
  orientation at the zero pose, which is used to find the ground-facing face and
  to align the resulting box with the world so it lies flat on the floor.

  Returns position, half-sizes and orientation, all in the link frame.
  """
  world = mesh.vertices @ rot.T
  z0 = world[:, 2].min()
  sole = world[world[:, 2] <= z0 + SOLE_BAND]
  lo, hi = sole.min(axis=0), sole.max(axis=0)

  size = np.array([(hi[0] - lo[0]) / 2.0, (hi[1] - lo[1]) / 2.0, SOLE_HALF_THICKNESS])
  centre_world = np.array(
    [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, z0 + SOLE_HALF_THICKNESS]
  )
  # Back into the link frame: the box's axes must map onto the world axes.
  return rot.T @ centre_world, size, mat_to_quat(rot.T)


def build_xml(
  links: dict[str, Link],
  joints: list[Joint],
  colliders: dict[str, str],
  foot_sites: dict[str, np.ndarray],
  spawn_z: float,
) -> str:
  by_parent: dict[str, list[Joint]] = {}
  for j in joints:
    by_parent.setdefault(j.parent, []).append(j)
  joint_to_child = {j.child: j for j in joints}

  def emit(link_name: str, depth: int) -> list[str]:
    pad = "  " * depth
    link = links[link_name]
    out = []

    if link_name == "torso":
      out.append(f'{pad}<body name="torso" pos="0 0 {spawn_z:g}" childclass="dash">')
    else:
      j = joint_to_child[link_name]
      out.append(
        f'{pad}<body name="{link_name}" pos="{fmt(j.pos)}" quat="{fmt(j.quat)}">'
      )

    out.append(
      f'{pad}  <inertial pos="{fmt(link.com)}" mass="{link.mass:.6g}" '
      f'fullinertia="{fmt(link.inertia, 8)}"/>'
    )
    if link_name == "torso":
      out.append(f'{pad}  <freejoint name="floating_base_joint"/>')
    else:
      j = joint_to_child[link_name]
      out.append(
        f'{pad}  <joint name="{j.name}" axis="{fmt(j.axis)}" '
        f'range="{j.lower:.6g} {j.upper:.6g}"/>'
      )
    if link.mesh:
      out.append(
        f'{pad}  <geom class="visual" name="{link_name}_visual" mesh="{link_name}"/>'
      )
    if link_name in colliders:
      out.append(f"{pad}  {colliders[link_name]}")
    if link_name in foot_sites:
      # Sole site, for the foot-clearance and foot-slip rewards and the per-foot
      # terrain height scan.
      out.append(
        f'{pad}  <site name="{link_name}" pos="{fmt(foot_sites[link_name])}"/>'
      )
    if link_name == "torso":
      out.append(f'{pad}  <site name="imu_in_torso" size="0.01" pos="0 0 0.19"/>')

    for child_joint in by_parent.get(link_name, []):
      out += emit(child_joint.child, depth + 1)
    out.append(f"{pad}</body>")
    return out

  meshes = "\n".join(
    f'    <mesh name="{n}" file="{n}.stl"/>'
    for n in sorted(links)
    if links[n].mesh is not None
  )
  body = "\n".join(emit("torso", 3))

  return f"""<mujoco model="dash_v2">
  <!--
    GENERATED by scripts/import_dash_urdf.py from the designer's
    robotDataPackage URDF. Do not hand-edit: re-run the script instead, so the
    next revision of the URDF imports the same way.

    18 DOF: 5 per leg (hip yaw/roll/pitch, knee pitch, ankle pitch) and 4 per
    arm (shoulder pitch/roll/yaw, elbow pitch).

    Axes, sign conventions, joint limits, masses and inertias are the
    designer's, untouched. What this file adds on top of the URDF:
      - `torso` freed with a freejoint; the URDF welds it to a `world` link,
        which MuJoCo would merge away along with its 11.4 kg.
      - `left_`/`right_` renamed to `l_`/`r_`, and knee/ankle/elbow suffixed
        `_pitch`, to match this repo's actuator and reward regexes.
      - Collision meshes (which the URDF ships as full-resolution copies of the
        visual meshes) replaced with fitted capsules and boxes, named
        `*_collision` for mjlab's CollisionCfg and contact sensors.
      - `l_foot`/`r_foot` sites at the soles, the IMU site, and the sensor block
        mjlab's velocity task reads by name.
  -->
  <compiler angle="radian" meshdir="assets_v2" autolimits="true"/>

  <default>
    <default class="dash">
      <default class="visual">
        <geom type="mesh" group="2" contype="0" conaffinity="0" density="0"/>
      </default>
      <default class="collision">
        <geom group="3" contype="1" conaffinity="1" density="0"/>
      </default>
    </default>
  </default>

  <asset>
{meshes}
  </asset>

  <worldbody>
{body}
  </worldbody>

  <!-- Pairs that a convex primitive cannot separate, excluded so the
       `self_collision` sensor does not report a permanent contact. MuJoCo only
       auto-excludes parent-child pairs, so these have to be listed.

       hip/upper_leg: padded to meet across the intervening dist_hip, so the
       capsules intersect at every pose. Same pair dash.xml excludes.

       lower_arm against torso/hip/dist_hip/upper_leg: the forearm swings down
       the side of the body through a gap that is real but narrow. Sampling
       poses within +-0.35 rad of the nominal stance, these pairs report capsule
       penetrations up to 74 mm at poses where the actual meshes still clear by
       3 to 47 mm -- every one is an artifact of wrapping two flat links in
       round capsules, not interference. Thinning cannot fix it: the false
       penetration exceeds the true gap, and these colliders need honest radii
       because `illegal_contact` uses them to catch limbs hitting the ground.
       Left unexcluded, `self_collisions` (weight -1.0) would fire through most
       of the arm swing and train the policy to hold its arms still. -->
  <contact>
    <exclude body1="r_hip" body2="r_upper_leg"/>
    <exclude body1="l_hip" body2="l_upper_leg"/>
    <exclude body1="r_lower_arm" body2="torso"/>
    <exclude body1="l_lower_arm" body2="torso"/>
    <exclude body1="r_lower_arm" body2="r_hip"/>
    <exclude body1="l_lower_arm" body2="l_hip"/>
    <exclude body1="r_lower_arm" body2="r_dist_hip"/>
    <exclude body1="l_lower_arm" body2="l_dist_hip"/>
    <exclude body1="r_lower_arm" body2="r_upper_leg"/>
    <exclude body1="l_lower_arm" body2="l_upper_leg"/>
  </contact>

  <!-- mjlab's velocity and tracking tasks read these by name: base_lin_vel and
       base_ang_vel observations, the upright reward, and the angular momentum
       penalty. Renaming any of them breaks those terms. -->
  <sensor>
    <gyro name="imu_ang_vel" site="imu_in_torso"/>
    <velocimeter name="imu_lin_vel" site="imu_in_torso"/>
    <accelerometer name="imu_lin_acc" site="imu_in_torso"/>
    <framezaxis name="imu_upvector" objtype="body" objname="world" reftype="site" refname="imu_in_torso"/>
    <subtreeangmom name="root_angmom" body="torso"/>
  </sensor>
</mujoco>
"""


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--src", type=Path, required=True, help="robotDataPackage root.")
  parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
  parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
  parser.add_argument("--target-faces", type=int, default=8000)
  parser.add_argument("--capsule-pct", type=float, default=CAPSULE_RADIUS_PCT)
  args = parser.parse_args()

  links, joints = parse_urdf(args.src / "urdf" / "Dash.urdf")
  print(f"parsed {len(links)} links, {len(joints)} revolute joints")
  poses = forward_kinematics(joints)

  # Collision primitives, fitted to the full-resolution collision meshes.
  colliders: dict[str, str] = {}
  soles: dict[str, float] = {}
  foot_sites: dict[str, np.ndarray] = {}
  for name, link in sorted(links.items()):
    if link.mesh is None:
      continue
    mesh = trimesh.load(args.src / "meshes" / "collision" / f"{link.mesh}.stl")
    stem = name[2:] if name[:2] in ("l_", "r_") else name
    if stem == "foot":
      rot, origin = poses[name]
      pos, size, quat = fit_sole(mesh, rot)
      colliders[name] = (
        f'<geom class="collision" name="{name}_collision" type="box" '
        f'pos="{fmt(pos)}" quat="{fmt(quat)}" size="{fmt(size)}"/>'
      )
      # World height of this sole at the zero pose, for the spawn height.
      soles[name] = origin[2] + (rot @ pos)[2] - size[2]
      foot_sites[name] = rot.T @ ((rot @ pos) - np.array([0.0, 0.0, size[2]]))
      print(f"  {name:16s} sole    size={fmt(size)} world_z={soles[name]:.4f}")
    elif stem in BOX_LINKS:
      pos, size = fit_box(mesh, TORSO_EXTENT_PCT)
      colliders[name] = (
        f'<geom class="collision" name="{name}_collision" type="box" '
        f'pos="{fmt(pos)}" size="{fmt(size)}"/>'
      )
      print(f"  {name:16s} box     size={fmt(size)}")
    elif stem in CAPSULE_LINKS:
      a, b, r = fit_capsule(mesh, args.capsule_pct)
      colliders[name] = (
        f'<geom class="collision" name="{name}_collision" type="capsule" '
        f'fromto="{fmt(np.concatenate([a, b]))}" size="{r:.6g}"/>'
      )
      print(f"  {name:16s} capsule r={r:.4f} len={np.linalg.norm(b - a):.4f}")

  # Spawn height puts the lowest sole on the ground at the zero pose. The
  # bent-knee keyframe in dash_constants shortens the legs and sets its own init
  # z on top of this; this is only the model's own resting height.
  spawn_z = -min(soles.values())
  print(f"spawn height (zero pose, soles on ground): {spawn_z:.6f} m")

  args.assets.mkdir(parents=True, exist_ok=True)
  before = after = 0
  for name, link in sorted(links.items()):
    if link.mesh is None:
      continue
    mesh = trimesh.load(args.src / "meshes" / "visual" / f"{link.mesh}.stl")
    n0 = len(mesh.faces)
    if n0 > args.target_faces:
      mesh = mesh.simplify_quadric_decimation(face_count=args.target_faces)
    mesh.export(args.assets / f"{name}.stl")
    before, after = before + n0, after + len(mesh.faces)
    print(f"  {link.mesh:24s} -> {name}.stl  {n0:7d} -> {len(mesh.faces):6d} faces")
  print(f"faces: {before} -> {after}")

  args.xml.write_text(build_xml(links, joints, colliders, foot_sites, spawn_z))
  print(f"wrote {args.xml}")


if __name__ == "__main__":
  main()
