# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL vec-env wrapper that sanitizes observations for cable-bend training."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def _sanitize_tensor(x: torch.Tensor, clip: float = 20.0) -> torch.Tensor:
    y = torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
    return torch.clamp(y, min=-clip, max=clip)


def _sanitize_tensordict(obs: TensorDict, clip: float = 20.0) -> TensorDict:
    out = {}
    for key, value in obs.items():
        out[key] = _sanitize_tensor(value, clip=clip)
    return TensorDict(out, batch_size=obs.batch_size)


class CableBendRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Clamp non-finite cable-bend observations before they reach PPO."""

    def __init__(self, env, clip_actions: float | None = None, obs_clip: float = 20.0):
        super().__init__(env, clip_actions=clip_actions)
        self._obs_clip = obs_clip

    def reset(self) -> tuple[TensorDict, dict]:
        obs, extras = super().reset()
        return _sanitize_tensordict(obs, clip=self._obs_clip), extras

    def get_observations(self) -> TensorDict:
        return _sanitize_tensordict(super().get_observations(), clip=self._obs_clip)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        obs, rew, dones, extras = super().step(actions)
        rew = torch.nan_to_num(rew, nan=0.0, posinf=100.0, neginf=-100.0)
        rew = torch.clamp(rew, -10.0, 10.0)
        return _sanitize_tensordict(obs, clip=self._obs_clip), rew, dones, extras
