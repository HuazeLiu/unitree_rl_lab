# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Assemble policy observation blocks matching training layout (vision path)."""

from __future__ import annotations

import torch

from unitree_rl_lab.assets.cable.u_shape_targets import NUM_U_DEPTH_PRESETS, depth_id_one_hot


def build_cable_privileged_obs_vector(
    endpoints_body: torch.Tensor,
    centerline_sparse_body: torch.Tensor,
    grasp_progress: torch.Tensor,
    depth_ids: torch.Tensor,
    phase_progress: torch.Tensor | None = None,
) -> torch.Tensor:
    """Concatenate cable-related policy obs (single env row).

    Layout matches training terms:
    ``cable_bend_command (4) + endpoints (6) + centerline (24) + grasp (2) + phase (1)``.

    Args:
        endpoints_body: Shape ``(6,)`` or ``(1, 6)``.
        centerline_sparse_body: Shape ``(24,)`` or ``(1, 24)``.
        grasp_progress: Shape ``(2,)`` or ``(1, 2)``.
        depth_ids: Shape ``()`` or ``(1,)``.
        phase_progress: Optional shape ``(1,)``.

    Returns:
        Tensor of shape ``(1, 37)``.
    """
    if endpoints_body.ndim == 1:
        endpoints_body = endpoints_body.unsqueeze(0)
    if centerline_sparse_body.ndim == 1:
        centerline_sparse_body = centerline_sparse_body.unsqueeze(0)
    if grasp_progress.ndim == 1:
        grasp_progress = grasp_progress.unsqueeze(0)
    if depth_ids.ndim == 0:
        depth_ids = depth_ids.unsqueeze(0)

    cmd = depth_id_one_hot(depth_ids, num_presets=NUM_U_DEPTH_PRESETS)
    if phase_progress is None:
        phase_progress = torch.zeros(1, 1, device=endpoints_body.device)
    elif phase_progress.ndim == 1:
        phase_progress = phase_progress.unsqueeze(0)

    return torch.cat(
        [cmd, endpoints_body, centerline_sparse_body, grasp_progress, phase_progress],
        dim=-1,
    )
