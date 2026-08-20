"""PPO configuration for the Dash upper-body pick-and-place task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def dash_lift_box_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO runner config, following mjlab's YAM lift-cube settings.

  The observation and action spaces here are smaller than YAM's (8 joints
  against 7, but no camera), and the task is not obviously harder to represent,
  so the network sizes and the algorithm block are left at mjlab's values. The
  iteration count is raised: a friction-only bimanual grip has a much narrower
  success basin than a parallel gripper, so the early phase spends far longer
  finding the first successful lift.
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
    experiment_name="dash_lift_box",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10_000,
  )
