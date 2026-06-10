# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-environment cable bend task state (grasp, success counters)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

CurriculumStage = Literal["bend", "full"]


@dataclass
class CableBendEnvState:
    """Mutable state for cable bend MDP helpers."""

    curriculum_stage: CurriculumStage = "bend"
    dual_grasp_steps: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    success_steps: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    cable_attached: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.bool))
    table_height: float = 0.75


_STATE: dict[int, CableBendEnvState] = {}


def init_cable_bend_state(env: ManagerBasedRLEnv, curriculum_stage: CurriculumStage, table_height: float) -> CableBendEnvState:
    """Create or refresh state buffers for an environment instance."""
    state = CableBendEnvState(
        curriculum_stage=curriculum_stage,
        dual_grasp_steps=torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
        success_steps=torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
        cable_attached=torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
        table_height=table_height,
    )
    _STATE[id(env)] = state
    return state


def get_cable_bend_state(env: ManagerBasedRLEnv) -> CableBendEnvState:
    """Return cable bend state, initializing with defaults if missing."""
    key = id(env)
    if key not in _STATE:
        stage: CurriculumStage = getattr(getattr(env, "cfg", None), "curriculum_stage", "bend")
        table_height = getattr(getattr(env, "cfg", None), "table_height", 0.75)
        return init_cable_bend_state(env, stage, table_height)
    return _STATE[key]


def reset_cable_bend_state(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice) -> None:
    """Reset per-env counters on episode reset."""
    state = get_cable_bend_state(env)
    # Clear the per-episode chamfer cache so bend_progress_reward gets a
    # fresh baseline on the first step of the new episode (prevents a
    # spurious progress spike from the last frame of the previous episode).
    if hasattr(state, "_prev_chamfer"):
        del state._prev_chamfer
    if isinstance(env_ids, slice):
        state.dual_grasp_steps[:] = 0
        state.success_steps[:] = 0
        if state.curriculum_stage == "full":
            state.cable_attached[env_ids] = False
        return
    state.dual_grasp_steps[env_ids] = 0
    state.success_steps[env_ids] = 0
    if state.curriculum_stage == "full":
        state.cable_attached[env_ids] = False
