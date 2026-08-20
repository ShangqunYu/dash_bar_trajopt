"""PPO configuration for the Dash upper-body bar-angle task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def dash_bar_angle_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO runner config, following the lift-box task's settings.

  The network and algorithm blocks are identical -- the observation and action
  spaces are close in size and the task is not harder to represent. The
  iteration budget is half the lift-box task's: any touch anywhere along the
  bar moves the angle and scores, so the exploration problem the box task's
  10k iterations were budgeted for (finding a two-handed squeeze that holds)
  does not exist here.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="dash_bar_angle",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=5_000,
  )
