import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

_ARMHOLD_VARIANTS = {
    "Unitree-G1-29dof-Velocity-ArmHold": ("ArmHoldEnvCfg", "ArmHoldPlayEnvCfg"),
    "Unitree-G1-29dof-Velocity-ArmHold-Guard": ("ArmHoldGuardEnvCfg", "ArmHoldGuardPlayEnvCfg"),
    "Unitree-G1-29dof-Velocity-ArmHold-Low": ("ArmHoldLowEnvCfg", "ArmHoldLowPlayEnvCfg"),
    "Unitree-G1-29dof-Velocity-ArmHold-Forward": ("ArmHoldForwardEnvCfg", "ArmHoldForwardPlayEnvCfg"),
    "Unitree-G1-29dof-Velocity-ArmHold-Tpose": ("ArmHoldTposeEnvCfg", "ArmHoldTposePlayEnvCfg"),
}

for task_id, (train_cfg, play_cfg) in _ARMHOLD_VARIANTS.items():
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.velocity_armhold_env_cfg:{train_cfg}",
            "play_env_cfg_entry_point": f"{__name__}.velocity_armhold_env_cfg:{play_cfg}",
            "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
        },
    )

gym.register(
    id="Unitree-G1-29dof-Cable-Hold",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_cable_hold_env_cfg:G1CableHoldEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.g1_cable_hold_env_cfg:G1CableHoldPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Cable-Bend-U",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_cable_bend_u_env_cfg:G1CableBendUEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.g1_cable_bend_u_env_cfg:G1CableBendUPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_cable_bend_ppo_cfg:CableBendUPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Cable-Bend-U-Full",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_cable_bend_u_env_cfg:G1CableBendUFullEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.g1_cable_bend_u_env_cfg:G1CableBendUPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_cable_bend_ppo_cfg:CableBendUPPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# Round 1 Experiment environments (short 300-iter sweep)
# ---------------------------------------------------------------------------
_EXP_VARIANTS: dict[str, tuple[str, str]] = {
    "Unitree-G1-29dof-Cable-Bend-U-Exp-Baseline": ("G1CableBendUExpBaselineCfg", "ExpBasePPORunnerCfg"),
    "Unitree-G1-29dof-Cable-Bend-U-Exp-1A":       ("G1CableBendUExp1ACfg",       "ExpBasePPORunnerCfg"),
    "Unitree-G1-29dof-Cable-Bend-U-Exp-1B":       ("G1CableBendUExp1BCfg",       "Exp1BPPORunnerCfg"),
    "Unitree-G1-29dof-Cable-Bend-U-Exp-1C":       ("G1CableBendUExp1CCfg",       "Exp1CPPORunnerCfg"),
    "Unitree-G1-29dof-Cable-Bend-U-Exp-1D":       ("G1CableBendUExp1DCfg",       "ExpBasePPORunnerCfg"),
}

for task_id, (env_cfg, ppo_cfg) in _EXP_VARIANTS.items():
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_cable_bend_u_env_cfg:{env_cfg}",
            "play_env_cfg_entry_point": f"{__name__}.g1_cable_bend_u_env_cfg:G1CableBendUExpBaselineCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_cable_bend_ppo_cfg:{ppo_cfg}",
        },
    )
