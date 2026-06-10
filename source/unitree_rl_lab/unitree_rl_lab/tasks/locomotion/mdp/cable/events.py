# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Domain randomisation and reset events for cable U-bend manipulation."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from unitree_rl_lab.assets.cable.g1_cable_attach import (
    attach_cable_endpoints_to_g1,
    reset_attachment_cache,
)
from unitree_rl_lab.assets.cable.thick_cable_cfg import STIFF_FLAT_TUBE_CFG, ThickCableCfg

from .state import get_cable_bend_state, init_cable_bend_state, reset_cable_bend_state

if True:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Startup  (run once at environment creation)
# ---------------------------------------------------------------------------


def setup_cable_bend_state(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    curriculum_stage: str = "bend",
    table_height: float = 0.75,
) -> None:
    """Initialise per-environment state buffers."""
    init_cable_bend_state(env, curriculum_stage=curriculum_stage, table_height=table_height)  # type: ignore[arg-type]


def clear_attachment_cache_on_startup(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
) -> None:
    """Clear fixed-joint attachment cache (idempotent)."""
    del env, env_ids
    reset_attachment_cache()


def weld_cable_on_prestartup(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    cable_hold_cfg: ThickCableCfg | None = None,
) -> None:
    """Create hand ↔ cable point welds BEFORE the GPU physics view is built."""
    del env_ids
    from unitree_rl_lab.assets.cable.g1_cable_attach import weld_cable_endpoints_buildtime

    weld_cable_endpoints_buildtime(env, cable_hold_cfg or STIFF_FLAT_TUBE_CFG)


def randomize_cable_physics(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    stiffness_scale: tuple[float, float] = (0.85, 1.15),
    mass_scale: tuple[float, float] = (0.9, 1.1),
) -> None:
    """Startup DR placeholder — cable params are fixed at spawn; hook retained."""
    del env, env_ids, stiffness_scale, mass_scale


# ---------------------------------------------------------------------------
# Reset  (run at the start of every episode)
# ---------------------------------------------------------------------------


def reset_cable_bend_episode_state(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
) -> None:
    """Clear per-episode counters and the progress-reward cache."""
    if env_ids is None:
        env_ids = slice(None)
    reset_cable_bend_state(env, env_ids)


def attach_g1_cable_on_reset(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    cable_hold_cfg: ThickCableCfg | None = None,
) -> None:
    """Align cable between hands and weld endpoints (bend-only curriculum)."""
    state = get_cable_bend_state(env)
    if state.curriculum_stage != "bend":
        return
    attach_cable_endpoints_to_g1(
        env, env_ids, cable_hold_cfg=cable_hold_cfg or STIFF_FLAT_TUBE_CFG
    )
    if env_ids is None:
        state.cable_attached[:] = True
    else:
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)
        state.cable_attached[env_ids] = True


def reset_cable_on_table(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    pose_range: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Stage-1 only: place cable straight on the table with small perturbation."""
    state = get_cable_bend_state(env)
    if state.curriculum_stage != "full":
        return

    cable: Articulation = env.scene[cable_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    pose_range = pose_range or {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.15, 0.15)}
    num = len(env_ids)
    device = env.device
    root = cable.data.default_root_state[env_ids].clone()

    x_off = torch.empty(num, device=device).uniform_(*pose_range["x"])
    y_off = torch.empty(num, device=device).uniform_(*pose_range["y"])
    yaw = torch.empty(num, device=device).uniform_(*pose_range["yaw"])

    height = state.table_height + STIFF_FLAT_TUBE_CFG.radius + 0.01
    root[:, 0] = x_off
    root[:, 1] = y_off
    root[:, 2] = height
    root[:, 3] = torch.cos(yaw * 0.5)
    root[:, 4] = 0.0
    root[:, 5] = 0.0
    root[:, 6] = torch.sin(yaw * 0.5)
    root[:, 7:] = 0.0

    cable.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
    cable.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)
    zero_joint = torch.zeros(num, cable.num_joints, device=device)
    cable.write_joint_state_to_sim(zero_joint, zero_joint, env_ids=env_ids)


# ---------------------------------------------------------------------------
# Interval  (run every N steps)
# ---------------------------------------------------------------------------


def try_attach_on_dual_grasp(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    grasp_threshold: float = 0.08,
    hold_steps: int = 10,
    cable_hold_cfg: ThickCableCfg | None = None,
) -> None:
    """Stage-1 only: weld cable to hands after sustained dual grasp."""
    state = get_cable_bend_state(env)
    if state.curriculum_stage != "full":
        return

    from .rewards import dual_grasp_reward

    grasp_score = dual_grasp_reward(env, grasp_threshold=grasp_threshold)
    close = grasp_score > 0.5
    state.dual_grasp_steps = torch.where(
        close, state.dual_grasp_steps + 1, torch.zeros_like(state.dual_grasp_steps)
    )
    ready = (state.dual_grasp_steps >= hold_steps) & (~state.cable_attached)
    if not torch.any(ready):
        return
    attach_ids = ready.nonzero(as_tuple=False).flatten()
    attach_cable_endpoints_to_g1(
        env, attach_ids, cable_hold_cfg=cable_hold_cfg or STIFF_FLAT_TUBE_CFG
    )
    state.cable_attached[attach_ids] = True


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def event_noop(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
) -> None:
    """No-op event placeholder for disabled event slots."""
    del env, env_ids
