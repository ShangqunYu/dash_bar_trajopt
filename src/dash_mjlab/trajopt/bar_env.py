"""Trajectory-optimization evaluation environment for the bar-angle task.

Not an RL environment. This is a deterministic black-box cost evaluator: a
collaborator hands over one vectorized control law -- phase in, right-arm
joint targets out -- and gets back a scalar cost, the angular
distance between where the bar was asked to point and where it points when the
clock runs out. Same law in, same cost out, every time: plain single-threaded
MuJoCo on CPU, no randomization, no torch.

The scene is the same robot and object as the RL task ``Mjlab-Bar-Angle-Dash-
UpperBody`` (the specs are shared, not copied), minus the RL stack: a flat
ship's-wheel of horizontal spokes on a single vertical hinge in front of the
robot (``num_spokes``, default 3; 1 recovers the original lone bar). Gravity
has no torque about the hinge, so the wheel stays wherever it is pushed, less
what the joint damping bleeds off. The hinge angle -- and so the cost --
measures spoke 0, the red one.

The cost is dense where the terminal angle alone would be flat: a rollout
whose end-effector never touches the wheel is additionally charged its
closest approach to the wheel (times ``reach_penalty_weight``), so
never-touching candidates are ordered by how close they came. Contact at any
point during the rollout removes the term entirely.

By default only the right arm is driven; ``arms="both"`` drives all eight
joints. The law's output columns map to ``env.active_joints``, in order:

  0  r_shoulder_pitch
  1  r_shoulder_roll
  2  r_shoulder_yaw
  3  r_elbow_pitch
  4-7 (arms="both" only) the same four on the left

The law works in normalized units, so a search never has to know the robot:
it takes a phase in [0, 1) (the fraction of the horizon elapsed) with a
trailing axis of size one, and returns one number per active joint in
[-1, 1]. The env
decodes those into radians -- 0 holds the spawn pose, +-1 reaches the joint's
upper/lower limit -- so the do-nothing law is the zero function and the
reachable set is exactly the cube. Values outside it are clipped.

The batch axes are the caller's: the env may query the whole horizon in one
call or one step at a time (see ``precompute``), and a properly vectorized law
cannot tell the difference.

Tracking is a plain PD torque law, kp * (desired - q) - kd * qd, clamped to
the 30 Nm effort limit. The default kp = 20 is deliberately soft -- gravity
sags the shoulder visibly below its commanded position -- so the trajectories
have to reason about dynamics, not just kinematics. With ``arms="right"`` the
left arm is PD-held at the spawn pose with the stiff gains the RL task trains
with; it never reaches the bar.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, overload

import mujoco
import numpy as np
from jaxtyping import Float

if TYPE_CHECKING:
    from mjviser import ViserMujocoScene

from dash_mjlab.robots.dash_upper_body_constants import (
    ARMS_READY_KEYFRAME,
    TORSO_MOUNT_HEIGHT,
    get_spec,
)
from dash_mjlab.tasks.bar_angle.env_cfgs import (
    BAR_HEIGHT,
    BAR_LENGTH,
    PIVOT_X,
    get_bar_spec,
)

ControlLaw = Callable[[Float[np.ndarray, "... 1"]], Float[np.ndarray, "... J"]]

ACTIVE_JOINTS: tuple[str, ...] = (
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_shoulder_yaw",
    "r_elbow_pitch",
)

_LEFT_JOINTS: tuple[str, ...] = (
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_shoulder_yaw",
    "l_elbow_pitch",
)

# The stiff hold on a parked arm, matching the RL task's actuator gains
# (dash_upper_body_constants): 200/6 on shoulder pitch and roll, 100/3 on
# shoulder yaw and elbow.
_HOLD_KP = np.array([200.0, 200.0, 100.0, 100.0])
_HOLD_KD = np.array([6.0, 6.0, 3.0, 3.0])

_EFFORT_LIMIT = 30.0
# Matches the RL entity's actuator armature, so the two setups share dynamics.
_ARMATURE = 0.01

# Spawn pose, same as the RL task's ready keyframe: arms slightly forward,
# elbows broken, hands straddling the bar's spawn line.
_SPAWN_POSE = {
    "shoulder_pitch": -0.3,
    "shoulder_roll": 0.0,
    "shoulder_yaw": 0.0,
    "elbow_pitch": -0.8,
}
assert ARMS_READY_KEYFRAME.joint_pos is not None  # Keep the two poses in sync.
assert (
    _SPAWN_POSE["shoulder_pitch"] == ARMS_READY_KEYFRAME.joint_pos[".*_shoulder_pitch"]
)
assert _SPAWN_POSE["elbow_pitch"] == ARMS_READY_KEYFRAME.joint_pos[".*_elbow_pitch"]


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# Offscreen rendering happens in a subprocess that imports only mujoco: torch
# (pulled in by mjlab) and OSMesa's software GL cannot share a process, so an
# in-process mujoco.Renderer segfaults on headless nodes.
_REPLAY_SRC = """
import sys

import imageio
import mujoco
import numpy as np

model_path, state_path, out_path, fps, height, width = sys.argv[1:7]
model = mujoco.MjModel.from_binary_path(model_path)
data = mujoco.MjData(model)
state = np.load(state_path)
data.mocap_pos[:] = state["mocap_pos"]
data.mocap_quat[:] = state["mocap_quat"]
renderer = mujoco.Renderer(model, height=int(height), width=int(width))
frames = []
for qpos in state["qpos"]:
  data.qpos[:] = qpos
  mujoco.mj_forward(model, data)
  renderer.update_scene(data, camera="video")
  frames.append(renderer.render())
imageio.mimwrite(out_path, frames, fps=int(fps))
renderer.close()
"""


def _render_video(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_frames: Float[np.ndarray, "F Q"],
    video_path: str,
    fps: int,
    size: tuple[int, int],
) -> None:
    """Replay recorded qpos frames to a video file in a torch-free subprocess."""
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "model.mjb")
        state_path = os.path.join(tmp, "state.npz")
        mujoco.mj_saveModel(model, model_path, None)
        np.savez(
            state_path,
            qpos=qpos_frames,
            mocap_pos=data.mocap_pos,
            mocap_quat=data.mocap_quat,
        )
        env = os.environ | {"MUJOCO_GL": os.environ.get("MUJOCO_GL", "osmesa")}
        subprocess.run(
            [
                sys.executable,
                "-c",
                _REPLAY_SRC,
                model_path,
                state_path,
                video_path,
                str(fps),
                str(size[0]),
                str(size[1]),
            ],
            check=True,
            env=env,
        )


def _build_model(timestep: float, num_spokes: int) -> mujoco.MjModel:
    """Robot at the origin, spoked wheel on its post at PIVOT_X, plus a ghost.

    The ghost is a mocap body carrying a translucent copy of spoke 0: rotating
    its quaternion displays the commanded angle in the viewer without touching
    the physics.
    """
    spec = get_spec()
    spec.option.timestep = timestep
    torso = spec.body("torso")
    torso.pos = np.array([0.0, 0.0, TORSO_MOUNT_HEIGHT])
    for joint in spec.joints:
        joint.armature = _ARMATURE

    frame = spec.worldbody.add_frame(pos=[PIVOT_X, 0.0, 0.0])
    spec.attach(get_bar_spec(num_spokes), frame=frame)

    ghost = spec.worldbody.add_body(
        name="target_ghost", mocap=True, pos=[PIVOT_X, 0.0, 0.0]
    )
    ghost.add_geom(
        name="target_ghost_visual",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0.0, 0.0, BAR_HEIGHT, -BAR_LENGTH, 0.0, BAR_HEIGHT],
        size=[0.01, 0.0, 0.0],
        rgba=(0.1, 0.8, 0.2, 0.4),
        group=2,
        contype=0,
        conaffinity=0,
        density=0.0,
    )

    # A floor: the RL scene gets one from mjlab's terrain config, here it has to
    # be explicit. Purely cosmetic-plus-safety -- nothing should ever reach it.
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05],
        rgba=(0.35, 0.38, 0.40, 1.0),
    )
    # Lighting: the compiled spec would otherwise only have the dim default
    # headlight, which renders video frames nearly black. One overhead light
    # with shadows for depth, one fill from the camera side.
    spec.worldbody.add_light(
        name="overhead",
        pos=[0.3, 0.0, 2.5],
        dir=[0.0, 0.0, -1.0],
        castshadow=True,
    )
    spec.worldbody.add_light(
        name="fill",
        pos=[1.5, -1.0, 1.5],
        dir=[-0.6, 0.45, -0.4],
        castshadow=False,
    )
    # Camera for offscreen video capture: from the robot's front-right, kept
    # aimed at the torso, which puts both the arm and the whole bar arc in
    # frame.
    spec.worldbody.add_camera(
        name="video",
        mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY,
        targetbody="torso",
        pos=[1.5, -1.1, 1.3],
    )
    return spec.compile()


class BarAngleTrajOptEnv:
    """Deterministic rollout evaluator for control-law candidates.

    The control law maps phase in [0, 1), shape ``(..., 1)``, to normalized
    joint targets in [-1, 1], shape ``(..., len(env.active_joints))`` -- 4 for
    the default ``arms="right"`` (r_shoulder_pitch, r_shoulder_roll,
    r_shoulder_yaw, r_elbow_pitch), 8 for ``arms="both"`` (the same four then
    their left counterparts). Zero holds the spawn pose; the env decodes to
    radians and clips. Typical use::

      def control_law(s):  # (..., 1) -> (..., 4)
        return np.concatenate(
          [shoulder_pitch(s), shoulder_roll(s), shoulder_yaw(s), elbow_pitch(s)],
          axis=-1,
        )

      env = BarAngleTrajOptEnv()
      cost = env.evaluate(control_law, target_angle=0.6)

    One instance can evaluate any number of candidates; each ``evaluate`` starts
    from the identical initial state. The instance is not thread-safe (it owns a
    single MjData); create one per worker for parallel search.
    """

    def __init__(
        self,
        horizon: float = 5.0,
        kp: float = 20.0,
        kd: float = 1.0,
        timestep: float = 0.005,
        initial_bar_angle: float = 0.0,
        num_spokes: int = 3,
        reach_penalty_weight: float = 1.0,
        arms: Literal["right", "both"] = "right",
    ):
        """See the class docstring. The shaping knobs:

        Args:
          arms: which arms the law drives. ``"right"`` (default) drives the four
            right-arm joints and stiff-holds the left arm at spawn; ``"both"``
            drives all eight, right columns first, with no held arm. The
            reaching term and contact test then cover both hands.
          num_spokes: spokes on the wheel. The default 3 makes it a flat ship's
            wheel -- a spoke is always within 60 degrees of the arm, so far more
            of the search space actually moves the object than with a single bar.
            Whatever the count, the spawn must stay contact-free (a spoke resting
            against a hand at reset nudges the wheel and makes "never touched"
            unreachable); with the current spawn pose, elbows at -0.8, counts up
            to at least 6 clear the parked hands by 10 cm or more.
            The hinge angle (and so the cost) still measures spoke 0, the red one.
          reach_penalty_weight: weight (rad per metre) of the dense reaching term.
            A rollout that never touches the wheel is scored
            ``angle_error + weight * min_distance(hand, wheel)``, so the flat
            never-touched plateau -- where every candidate used to cost exactly
            ``|target_angle|`` -- gains a slope pointing at the wheel. Any rollout
            that makes contact drops the term entirely; it never trades off
            against the angle. Set 0 to recover the pure terminal cost.
        """
        self.horizon = horizon
        self.kp = kp
        self.kd = kd
        self.timestep = timestep
        self.initial_bar_angle = initial_bar_angle
        self.num_spokes = num_spokes
        self.reach_penalty_weight = reach_penalty_weight
        self.arms = arms
        if arms not in ("right", "both"):
            raise ValueError(f"arms must be 'right' or 'both', got {arms!r}")
        self.active_joints: tuple[str, ...] = ACTIVE_JOINTS + (
            _LEFT_JOINTS if arms == "both" else ()
        )
        passive_joints: tuple[str, ...] = _LEFT_JOINTS if arms == "right" else ()

        self.model = _build_model(timestep, num_spokes)
        self.data = mujoco.MjData(self.model)

        def qadr(name: str) -> int:
            return self.model.joint(name).qposadr[0]

        def vadr(name: str) -> int:
            return self.model.joint(name).dofadr[0]

        self._active_qadr = np.array([qadr(n) for n in self.active_joints])
        self._active_vadr = np.array([vadr(n) for n in self.active_joints])
        self._active_range = np.array(
            [self.model.joint(n).range for n in self.active_joints]
        )  # (J, 2)
        self._passive_qadr = np.array([qadr(n) for n in passive_joints], dtype=int)
        self._passive_vadr = np.array([vadr(n) for n in passive_joints], dtype=int)
        self._passive_kp = _HOLD_KP[: len(passive_joints)]
        self._passive_kd = _HOLD_KD[: len(passive_joints)]
        # Attaching the bar spec prefixes its names (e.g. "/bar_joint"), so find
        # the joint by suffix rather than assuming the exact prefix convention.
        (bar_joint,) = [
            self.model.joint(i).name
            for i in range(self.model.njnt)
            if self.model.joint(i).name.endswith("bar_joint")
        ]
        self._bar_qadr = qadr(bar_joint)
        self._bar_vadr = vadr(bar_joint)

        def spawn(names: tuple[str, ...]) -> np.ndarray:
            return np.array([_SPAWN_POSE[n.split("_", 1)[1]] for n in names])

        self._active_spawn = spawn(self.active_joints)
        self._passive_spawn = spawn(passive_joints)
        self._ghost_mocap_id = self.model.body("target_ghost").mocapid[0]

        # Ids for the reaching term: the active arms' end-effector sites and
        # geoms, and the wheel's spoke geoms and tip sites (attach-prefixed,
        # hence the suffix matching). The pivot is fixed in the world, so
        # hand-to-wheel distance is min over hands and spokes of
        # point-to-segment(pivot, tip_i).
        hands = ("r",) if arms == "right" else ("r", "l")
        self._hand_site_ids = np.array([self.model.site(f"{h}_hand").id for h in hands])
        self._hand_geom_ids = frozenset(
            self.model.geom(f"{h}_lower_arm_collision").id for h in hands
        )

        def _suffix_ids(kind: str, names: list[str], suffixes: list[str]) -> np.ndarray:
            ids = []
            for suffix in suffixes:
                (match,) = [n for n in names if n.endswith(suffix)]
                ids.append(getattr(self.model, kind)(match).id)
            return np.array(ids)

        spoke_suffixes = [
            "bar_tip" if i == 0 else f"bar_tip_{i}" for i in range(num_spokes)
        ]
        site_names = [self.model.site(i).name for i in range(self.model.nsite)]
        self._spoke_tip_site_ids = _suffix_ids("site", site_names, spoke_suffixes)
        geom_suffixes = [
            "bar_geom" if i == 0 else f"bar_geom_{i}" for i in range(num_spokes)
        ]
        geom_names = [self.model.geom(i).name for i in range(self.model.ngeom)]
        self._spoke_geom_ids = frozenset(
            int(g) for g in _suffix_ids("geom", geom_names, geom_suffixes)
        )
        self._pivot_pos = np.array([PIVOT_X, 0.0, BAR_HEIGHT])
        # Lazily created on the first render and reused after, so repeated
        # rendered evaluations stream into the same browser tab.
        self._viser_scene: ViserMujocoScene | None = None

    @property
    def num_steps(self) -> int:
        return round(self.horizon / self.timestep)

    def bar_angle(self) -> float:
        """Current bar hinge angle (rad). 0 points spoke 0 at the robot, positive
        swings its free end to the robot's right."""
        return float(self.data.qpos[self._bar_qadr])

    def _hand_to_wheel_distance(self) -> float:
        """Distance (m) from the nearest active end-effector site to the nearest
        spoke's centreline segment (pivot to tip). Surface offsets (capsule
        radii) are a constant the min cannot see, so the segment distance is the
        right shape."""
        hands = self.data.site_xpos[self._hand_site_ids]  # (H, 3)
        tips = self.data.site_xpos[self._spoke_tip_site_ids]  # (num_spokes, 3)
        seg = tips - self._pivot_pos
        rel = hands - self._pivot_pos
        t = np.clip(
            (rel @ seg.T) / np.einsum("ij,ij->i", seg, seg), 0.0, 1.0
        )  # (H, num_spokes)
        closest = self._pivot_pos + t[..., None] * seg
        return float(np.linalg.norm(hands[:, None] - closest, axis=-1).min())

    def _hand_touching_wheel(self) -> bool:
        """True while any active end-effector geom is in contact with a spoke."""
        con = self.data.contact
        for i in range(self.data.ncon):
            g1, g2 = int(con.geom1[i]), int(con.geom2[i])
            if (g1 in self._hand_geom_ids and g2 in self._spoke_geom_ids) or (
                g2 in self._hand_geom_ids and g1 in self._spoke_geom_ids
            ):
                return True
        return False

    def _reset(self, target_angle: float) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._active_qadr] = self._active_spawn
        self.data.qpos[self._passive_qadr] = self._passive_spawn
        self.data.qpos[self._bar_qadr] = self.initial_bar_angle
        # Point the ghost at the target: the ghost geom is a copy of the bar's, so
        # the same hinge angle is a pure z rotation.
        self.data.mocap_quat[self._ghost_mocap_id] = [
            math.cos(target_angle / 2.0),
            0.0,
            0.0,
            math.sin(target_angle / 2.0),
        ]
        mujoco.mj_forward(self.model, self.data)

    def _decode(self, action: Float[np.ndarray, "M J"]) -> Float[np.ndarray, "M J"]:
        """Normalized joint targets in [-1, 1] -> radians: 0 is the spawn pose,
        +-1 the joint's upper/lower limit. Piecewise linear, since the spawn pose
        is not centred in its range."""
        action = np.clip(action, -1.0, 1.0)
        reach_up = self._active_range[:, 1] - self._active_spawn
        reach_down = self._active_spawn - self._active_range[:, 0]
        return self._active_spawn + action * np.where(
            action >= 0.0, reach_up, reach_down
        )

    def _query(
        self, control_law: ControlLaw, times: Float[np.ndarray, " M"]
    ) -> Float[np.ndarray, "M J"]:
        """Desired joint positions (rad) at `times`, decoded from the law's
        normalized output."""
        phases = times[:, None] / self.horizon
        action = np.asarray(control_law(phases), dtype=float)
        expected = (len(times), len(self.active_joints))
        if action.shape != expected:
            raise ValueError(
                f"Control law given phases of shape {phases.shape} must return "
                f"shape {expected} ({', '.join(self.active_joints)}), "
                f"got {action.shape}."
            )
        if not np.isfinite(action).all():
            bad = phases[~np.isfinite(action).all(axis=-1)][0, 0]
            raise ValueError(
                f"Control law returned a non-finite value at phase {bad:.3f}."
            )
        return self._decode(action)

    def _get_viser_scene(self) -> ViserMujocoScene:
        if self._viser_scene is None:
            import viser
            from mjviser import ViserMujocoScene

            # The server announces its own URL (default http://localhost:8080) and
            # stays up for the lifetime of this env instance.
            server = viser.ViserServer(label="dash-bar-trajopt")
            self._viser_scene = ViserMujocoScene(server, self.model, num_envs=1)
        return self._viser_scene

    @overload
    def evaluate(
        self,
        control_law: ControlLaw,
        target_angle: float,
        *,
        precompute: bool = ...,
        return_trajectory: Literal[False] = ...,
        render: bool = ...,
        render_backend: Literal["viser", "native"] = ...,
        video_path: str | None = ...,
        video_fps: int = ...,
        video_size: tuple[int, int] = ...,
    ) -> float: ...

    @overload
    def evaluate(
        self,
        control_law: ControlLaw,
        target_angle: float,
        *,
        precompute: bool = ...,
        return_trajectory: Literal[True],
        render: bool = ...,
        render_backend: Literal["viser", "native"] = ...,
        video_path: str | None = ...,
        video_fps: int = ...,
        video_size: tuple[int, int] = ...,
    ) -> tuple[float, dict[str, np.ndarray]]: ...

    def evaluate(
        self,
        control_law: ControlLaw,
        target_angle: float,
        *,
        precompute: bool = True,
        return_trajectory: bool = False,
        render: bool = False,
        render_backend: Literal["viser", "native"] = "viser",
        video_path: str | None = None,
        video_fps: int = 30,
        video_size: tuple[int, int] = (480, 640),
    ) -> float | tuple[float, dict[str, np.ndarray]]:
        """Roll out the control law and return the terminal cost.

        Args:
          control_law: vectorized map from phase (shape ``(..., 1)``, 0 to 1) to
            normalized joint targets (shape ``(..., len(env.active_joints))``,
            -1 to 1) in ``env.active_joints`` order. Zero holds the spawn pose, +-1 is the joint's
            limit; values outside the cube are clipped, non-finite raise.
          target_angle: commanded bar angle in rad.
          precompute: query the law once for the whole horizon before stepping.
            The law is a function of time alone, so this is equivalent to asking
            it step by step and saves ``num_steps - 1`` calls. Set False to have
            it queried one step at a time, which is what a state-dependent law
            would need.
          return_trajectory: also return the full rollout -- keys ``time``,
            ``bar_angle``, ``joint_pos``, ``joint_target``, ``hand_to_wheel``,
            and the scalar ``touched`` -- for debugging a candidate. Off by
            default so the search loop pays nothing for it.
          render: play the rollout at real time in a viewer. For watching single
            candidates, not for use inside a search loop.
          render_backend: ``"viser"`` (default) serves the scene to the browser --
            the server starts on the first rendered call, prints its URL, and is
            reused by later calls, so keep the process alive while watching. If no
            browser is connected yet, the rollout waits for one so the animation
            is not played into an empty room. ``"native"`` opens a MuJoCo window
            instead (needs a local display).
          video_path: if set, record the rollout offscreen to this file (e.g.
            ``"push.mp4"``). Independent of ``render``, runs at full speed, and
            works headless. Frames come from the built-in "video" camera.
          video_fps: frame rate of the written video.
          video_size: (height, width) of the written video.

        Returns:
          The cost: |shortest angular distance between target and final bar
          angle| in radians, plus -- only if the end-effector never contacted the
          wheel -- ``reach_penalty_weight`` times the closest approach (m) of the
          end-effector to the wheel over the rollout. With ``return_trajectory``,
          a ``(cost, trajectory)`` tuple instead.
        """
        self._reset(target_angle)

        n = self.num_steps
        # Always allocated: a few float arrays per rollout are noise next to the
        # physics, and it keeps every variable unconditionally bound.
        times = np.arange(n) * self.timestep
        schedule = self._query(control_law, times) if precompute else None
        bar_angles = np.empty(n)
        joint_pos = np.empty((n, len(self.active_joints)))
        joint_target = np.empty((n, len(self.active_joints)))
        hand_to_wheel = np.empty(n)
        touched = False

        viewer = None
        scene = None
        if render:
            if render_backend == "viser":
                scene = self._get_viser_scene()
                # Don't roll out into an empty room: without this, the animation has
                # already played by the time the user opens the printed URL and all
                # they ever see is the final pose.
                if not scene.server.get_clients():
                    print(
                        "Waiting for a browser to connect before starting the rollout..."
                    )
                    while not scene.server.get_clients():
                        time.sleep(0.1)
                    # Give the page a beat to finish loading the meshes.
                    time.sleep(1.0)
                scene.update_from_mjdata(self.data)
            elif render_backend == "native":
                from mujoco import viewer as mujoco_viewer

                viewer = mujoco_viewer.launch_passive(self.model, self.data)
            else:
                raise ValueError(f"Unknown render_backend: {render_backend!r}")

        # Rendering happens after the rollout in a torch-free subprocess (see
        # _render_video); here only the poses to replay are collected.
        qpos_frames: list[np.ndarray] = []

        try:
            for k in range(n):
                t = times[k]
                if schedule is not None:
                    desired = schedule[k]
                else:
                    desired = self._query(control_law, times[k : k + 1])[0]

                q = self.data.qpos[self._active_qadr]
                qd = self.data.qvel[self._active_vadr]
                tau_active = self.kp * (desired - q) - self.kd * qd
                q_p = self.data.qpos[self._passive_qadr]
                qd_p = self.data.qvel[self._passive_vadr]
                tau_passive = (
                    self._passive_kp * (self._passive_spawn - q_p)
                    - self._passive_kd * qd_p
                )

                self.data.qfrc_applied[self._active_vadr] = np.clip(
                    tau_active, -_EFFORT_LIMIT, _EFFORT_LIMIT
                )
                self.data.qfrc_applied[self._passive_vadr] = np.clip(
                    tau_passive, -_EFFORT_LIMIT, _EFFORT_LIMIT
                )
                mujoco.mj_step(self.model, self.data)

                bar_angles[k] = self.bar_angle()
                joint_pos[k] = self.data.qpos[self._active_qadr]
                joint_target[k] = desired
                hand_to_wheel[k] = self._hand_to_wheel_distance()
                touched = touched or self._hand_touching_wheel()
                if video_path is not None and t >= len(qpos_frames) / video_fps:
                    qpos_frames.append(self.data.qpos.copy())
                if scene is not None:
                    # ~66 Hz scene sync is enough for the browser; the sleep paces the
                    # whole rollout to real time.
                    if k % 3 == 0:
                        scene.update_from_mjdata(self.data)
                    time.sleep(self.timestep)
                elif viewer is not None:
                    if not viewer.is_running():
                        break
                    viewer.sync()
                    time.sleep(self.timestep)
            if scene is not None:
                scene.update_from_mjdata(self.data)  # Show the final settled state.
            if video_path is not None:
                _render_video(
                    self.model,
                    self.data,
                    np.array(qpos_frames),
                    video_path,
                    video_fps,
                    video_size,
                )
        finally:
            if viewer is not None:
                viewer.close()

        cost = abs(_wrap_to_pi(target_angle - self.bar_angle()))
        # Dense shaping for the never-touched plateau: without it every rollout
        # that misses the wheel costs exactly |target_angle| and a search gets no
        # gradient until it stumbles into contact. Conditional on contact rather
        # than always-on so a touching candidate is judged purely on the angle.
        if not touched:
            cost += self.reach_penalty_weight * float(hand_to_wheel.min())
        if return_trajectory:
            return cost, {
                "time": times,
                "bar_angle": bar_angles,
                "joint_pos": joint_pos,
                "joint_target": joint_target,
                "hand_to_wheel": hand_to_wheel,
                "touched": np.array(touched),
            }
        return cost
