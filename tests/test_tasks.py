"""Checks that the Dash tasks register and build."""

import pytest
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

import dash_mjlab.tasks  # noqa: F401  Populates the registry.

DASH_TASKS = [
  "Mjlab-Velocity-Flat-Dash",
  "Mjlab-Velocity-Rough-Dash",
  "Mjlab-Tracking-Flat-Dash",
  "Mjlab-Tracking-Flat-Dash-No-State-Estimation",
  "Mjlab-Lift-Box-Dash-UpperBody",
  "Mjlab-Bar-Angle-Dash-UpperBody",
]


def test_tasks_registered() -> None:
  registered = set(list_tasks())
  assert set(DASH_TASKS) <= registered


@pytest.mark.parametrize("task_id", DASH_TASKS)
def test_configs_load(task_id: str) -> None:
  env_cfg = load_env_cfg(task_id)
  load_env_cfg(task_id, play=True)
  load_rl_cfg(task_id)
  assert "robot" in env_cfg.scene.entities


def test_lift_box_scene_is_complete() -> None:
  """The manipulation task needs its object and its table, and the commands and
  rewards address them by name."""
  env_cfg = load_env_cfg("Mjlab-Lift-Box-Dash-UpperBody")
  assert set(env_cfg.scene.entities) == {"robot", "box", "table"}
  assert env_cfg.commands is not None
  lift_cmd = env_cfg.commands["lift_height"]
  assert isinstance(lift_cmd, LiftingCommandCfg)
  assert lift_cmd.entity_name == "box"
  assert env_cfg.rewards["lift"].params["object_name"] == "box"
