"""Dash velocity tracking task registration."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import dash_flat_env_cfg, dash_rough_env_cfg
from .rl_cfg import dash_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Dash",
  env_cfg=dash_rough_env_cfg(),
  play_env_cfg=dash_rough_env_cfg(play=True),
  rl_cfg=dash_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Dash",
  env_cfg=dash_flat_env_cfg(),
  play_env_cfg=dash_flat_env_cfg(play=True),
  rl_cfg=dash_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Same two tasks against the model imported from the designer's URDF. Separate
# task ids rather than a flag on the existing ones so the policies trained
# against dash.xml stay reproducible: the v2 robot has different masses, link
# geometry, joint limits and several joint sign conventions, so a checkpoint
# from one does not transfer to the other.
register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Dash-V2",
  env_cfg=dash_rough_env_cfg(v2=True),
  play_env_cfg=dash_rough_env_cfg(play=True, v2=True),
  rl_cfg=dash_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Dash-V2",
  env_cfg=dash_flat_env_cfg(v2=True),
  play_env_cfg=dash_flat_env_cfg(play=True, v2=True),
  rl_cfg=dash_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
