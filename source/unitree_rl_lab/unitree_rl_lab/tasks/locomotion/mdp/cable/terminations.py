# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for cable U-bend task."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from unitree_rl_lab.assets.cable.thick_cable_utils import get_cable_centerline_points

from .commands import get_cable_bend_command
from .rewards import update_success_counters
from .state import get_cable_bend_state

if True:
    from isaaclab.envs import ManagerBasedRLEnv


def cable_state_invalid(
    env: ManagerBasedRLEnv,
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Terminate when cable centerline contains non-finite values."""
    cable: Articulation = env.scene[cable_cfg.name]
    centerline = get_cable_centerline_points(cable)
    finite = torch.isfinite(centerline).all(dim=-1).all(dim=-1)
    return ~finite


def cable_bend_success(
    env: ManagerBasedRLEnv,
    command_name: str = "cable_bend_u",
    cable_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    success_threshold: float = 0.85,
    hold_steps: int = 20,
) -> torch.Tensor:
    """Terminate on sustained U-shape match (~0.67 s at 30 Hz control)."""
    update_success_counters(env, command_name, cable_cfg, success_threshold)
    state = get_cable_bend_state(env)
    return state.success_steps >= hold_steps
