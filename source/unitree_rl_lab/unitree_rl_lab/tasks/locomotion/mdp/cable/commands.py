# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Command generator for random U-shape bend depth targets."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.cable.u_shape_targets import (
    NUM_U_DEPTH_PRESETS,
    U_SHAPE_DEPTH_PRESETS,
    depth_id_one_hot,
    generate_u_shape_centerline_batch,
    get_u_shape_endpoint_targets,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class CableBendUCommand(CommandTerm):
    """Sample one of several U-depth presets per episode and cache target geometry."""

    cfg: CableBendUCommandCfg

    def __init__(self, cfg: CableBendUCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        torso_ids, _ = self.robot.find_bodies(cfg.torso_body_name)
        if not torso_ids:
            raise RuntimeError(f"Could not find torso body '{cfg.torso_body_name}' on robot.")
        self._torso_body_id = torso_ids[0]

        self.depth_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_centerline = torch.zeros(
            self.num_envs, cfg.num_target_points, 3, device=self.device
        )
        self.target_endpoint_poses: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = (
            torch.zeros(self.num_envs, 3, device=self.device),
            torch.zeros(self.num_envs, 4, device=self.device),
            torch.zeros(self.num_envs, 3, device=self.device),
            torch.zeros(self.num_envs, 4, device=self.device),
        )
        for q in self.target_endpoint_poses[1::2]:
            q[:, 0] = 1.0

        self.metrics["depth_id"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """One-hot U-depth command for the policy, shape ``(num_envs, num_presets)``."""
        return depth_id_one_hot(self.depth_ids)

    @property
    def depth_id_normalized(self) -> torch.Tensor:
        """Normalized depth in [0, 1], shape ``(num_envs, 1)``."""
        if NUM_U_DEPTH_PRESETS <= 1:
            return torch.zeros(self.num_envs, 1, device=self.device)
        return (self.depth_ids.float() / float(NUM_U_DEPTH_PRESETS - 1)).unsqueeze(1)

    def _update_metrics(self):
        self.metrics["depth_id"] = self.depth_ids.float()

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)

        if self.cfg.fixed_preset_id is not None:
            self.depth_ids[env_ids] = self.cfg.fixed_preset_id
        else:
            self.depth_ids[env_ids] = torch.randint(
                0, NUM_U_DEPTH_PRESETS, (len(env_ids),), device=self.device
            )

        torso_pos = self.robot.data.body_pos_w[env_ids, self._torso_body_id]
        torso_quat = self.robot.data.body_quat_w[env_ids, self._torso_body_id]
        targets = generate_u_shape_centerline_batch(
            self.depth_ids[env_ids],
            torso_pos,
            torso_quat,
            workspace_offset_body=self.cfg.workspace_offset_body,
            num_points=self.cfg.num_target_points,
        )
        self.target_centerline[env_ids] = targets
        start_p, start_q, end_p, end_q = get_u_shape_endpoint_targets(targets)
        self.target_endpoint_poses[0][env_ids] = start_p
        self.target_endpoint_poses[1][env_ids] = start_q
        self.target_endpoint_poses[2][env_ids] = end_p
        self.target_endpoint_poses[3][env_ids] = end_q

    def _update_command(self):
        pass


@configclass
class CableBendUCommandCfg(CommandTermCfg):
    """Configuration for the cable U-bend depth command."""

    class_type: type = CableBendUCommand

    asset_name: str = "robot"
    torso_body_name: str = "torso_link"
    num_target_points: int = 20
    # Origin of the U workspace in torso frame: forward to ~hand reach, slightly above torso so
    # the U endpoints sit near the held-cable hand height.
    workspace_offset_body: tuple[float, float, float] = (0.30, 0.0, 0.10)
    # If set, always use this preset ID (0=shallow, 1=med_shallow, ...); if None, randomize.
    fixed_preset_id: int | None = None

    def __post_init__(self):
        # Resample only on episode reset (not mid-episode); large value avoids mid-episode resampling.
        self.resampling_time_range = (1.0e9, 1.0e9)


def get_cable_bend_command(env: ManagerBasedRLEnv, command_name: str = "cable_bend_u") -> CableBendUCommand:
    """Return the cable bend command term instance."""
    return env.command_manager.get_term(command_name)  # type: ignore[return-value]
