# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""PPO runner configuration for the G1 cable U-bend task.

Tuned for 17-DOF bimanual manipulation with a stiff articulated cable.
Network: [512, 256, 128] ELU for both actor and critic.
Fixed learning rate, low entropy, mild action noise — conservative
hyperparameters that produce stable, monotonic convergence.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class CableBendUPPORunnerCfg(BasePPORunnerCfg):
    """Production PPO config for G1 cable U-bend with stiff articulated cable."""

    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 200
    experiment_name = "unitree_g1_29dof_cable_bend_u"
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.08,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=8.0e-4,
        schedule="",                     # fixed LR — adaptive can collapse
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
