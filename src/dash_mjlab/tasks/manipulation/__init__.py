"""Dash upper-body bimanual pick-and-place task registration."""

from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import dash_lift_box_env_cfg
from .rl_cfg import dash_lift_box_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Lift-Box-Dash-UpperBody",
  env_cfg=dash_lift_box_env_cfg(),
  play_env_cfg=dash_lift_box_env_cfg(play=True),
  rl_cfg=dash_lift_box_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
