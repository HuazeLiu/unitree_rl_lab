# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for thick cable state extraction and RL rewards."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation


def _segment_sort_key(name: str) -> int:
    if "seg_" in name:
        return int(name.rsplit("_", 1)[-1])
    return 0


def _resolve_body_ids(cable: Articulation, body_names: list[str] | None = None) -> list[int]:
    if body_names is None:
        return sorted(range(cable.num_bodies), key=lambda i: _segment_sort_key(cable.body_names[i]))
    body_ids: list[int] = []
    for name in body_names:
        ids, _ = cable.find_bodies(name)
        body_ids.extend(ids)
    return body_ids


def get_cable_segment_poses(
    cable: Articulation,
    env_ids: torch.Tensor | slice | None = None,
    body_names: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return segment positions and orientations (w, x, y, z).

    Returns:
        positions: Shape ``(num_envs, num_segments, 3)``.
        orientations: Shape ``(num_envs, num_segments, 4)``.
    """
    if env_ids is None:
        env_ids = slice(None)
    body_ids = _resolve_body_ids(cable, body_names)
    positions = cable.data.body_pos_w[env_ids][:, body_ids, :]
    orientations = cable.data.body_quat_w[env_ids][:, body_ids, :]
    return positions, orientations


def get_cable_centerline_points(
    cable: Articulation,
    env_ids: torch.Tensor | slice | None = None,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Return segment-center positions along the cable centerline.

    Returns:
        Tensor of shape ``(num_envs, num_segments, 3)``.
    """
    positions, _ = get_cable_segment_poses(cable, env_ids=env_ids, body_names=body_names)
    return positions


def get_cable_endpoint_poses(
    cable: Articulation,
    env_ids: torch.Tensor | slice | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return poses of the first and last cable segments.

    Returns:
        start_pos, start_quat, end_pos, end_quat with shapes ``(num_envs, 3)`` and ``(num_envs, 4)``.
    """
    positions, orientations = get_cable_segment_poses(cable, env_ids=env_ids)
    return positions[:, 0], orientations[:, 0], positions[:, -1], orientations[:, -1]


def compute_centerline_chamfer_reward(
    cable: Articulation,
    target_curve_points: torch.Tensor,
    env_ids: torch.Tensor | slice | None = None,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Soft chamfer-like reward between cable centerline and a target polyline.

    Args:
        cable: Cable articulation asset.
        target_curve_points: Target points with shape ``(num_envs, num_target, 3)`` or ``(num_target, 3)``.
        env_ids: Optional environment subset.
        temperature: Softmin temperature in meters.

    Returns:
        Reward in ``[0, 1]`` with shape ``(num_envs,)``.
    """
    centerline = get_cable_centerline_points(cable, env_ids=env_ids)
    if target_curve_points.ndim == 2:
        target_curve_points = target_curve_points.unsqueeze(0).expand(centerline.shape[0], -1, -1)

    # cable -> target
    dists_ct = torch.cdist(centerline, target_curve_points)
    min_ct = dists_ct.min(dim=-1).values
    soft_ct = -temperature * torch.logsumexp(-min_ct / temperature, dim=-1)

    # target -> cable
    dists_tc = dists_ct.min(dim=1).values
    soft_tc = -temperature * torch.logsumexp(-dists_tc / temperature, dim=-1)

    chamfer = 0.5 * (soft_ct + soft_tc)
    reward = torch.exp(-chamfer / temperature)
    return torch.clamp(reward, 0.0, 1.0)


def compute_endpoint_reward(
    cable: Articulation,
    target_endpoint_poses: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    env_ids: torch.Tensor | slice | None = None,
    pos_weight: float = 1.0,
    rot_weight: float = 0.25,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Reward for matching cable endpoint positions and orientations.

    Tries both orderings (start→start+end→end  AND  start→end+end→start) and returns the
    higher reward, so the signal is robust against the cable being attached left-to-right or
    right-to-left relative to the target curve orientation.

    Args:
        target_endpoint_poses: ``(start_pos, start_quat, end_pos, end_quat)`` same layout as
            :func:`get_cable_endpoint_poses`.
    """
    start_pos, start_quat, end_pos, end_quat = get_cable_endpoint_poses(cable, env_ids=env_ids)
    tgt_start_pos, tgt_start_quat, tgt_end_pos, tgt_end_quat = target_endpoint_poses

    # Forward ordering: cable-start ↔ target-start, cable-end ↔ target-end
    pos_err_fwd = torch.norm(start_pos - tgt_start_pos, dim=-1) + torch.norm(end_pos - tgt_end_pos, dim=-1)
    rot_err_fwd = _quat_geodesic_distance(start_quat, tgt_start_quat) + _quat_geodesic_distance(end_quat, tgt_end_quat)
    err_fwd = pos_weight * pos_err_fwd + rot_weight * rot_err_fwd

    # Reversed ordering: cable-start ↔ target-end, cable-end ↔ target-start
    pos_err_rev = torch.norm(start_pos - tgt_end_pos, dim=-1) + torch.norm(end_pos - tgt_start_pos, dim=-1)
    rot_err_rev = _quat_geodesic_distance(start_quat, tgt_end_quat) + _quat_geodesic_distance(end_quat, tgt_start_quat)
    err_rev = pos_weight * pos_err_rev + rot_weight * rot_err_rev

    best_err = torch.min(err_fwd, err_rev)
    return torch.exp(-best_err / temperature)


def compute_bending_energy(cable: Articulation, env_ids: torch.Tensor | slice | None = None) -> torch.Tensor:
    """Proxy bending energy as sum of squared joint positions."""
    if env_ids is None:
        env_ids = slice(None)
    joint_pos = cable.data.joint_pos[env_ids]
    return torch.sum(joint_pos * joint_pos, dim=-1)


def compute_smoothness_reward(
    cable: Articulation,
    env_ids: torch.Tensor | slice | None = None,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Reward smooth centerline curvature via second-difference penalty."""
    centerline = get_cable_centerline_points(cable, env_ids=env_ids)
    if centerline.shape[1] < 3:
        batch = centerline.shape[0]
        return torch.ones(batch, device=centerline.device)

    second_diff = centerline[:, 2:] - 2.0 * centerline[:, 1:-1] + centerline[:, :-2]
    curvature_proxy = torch.sum(second_diff * second_diff, dim=(1, 2))
    return torch.exp(-curvature_proxy / temperature)


def _quat_geodesic_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Geodesic angle between unit quaternions (w, x, y, z)."""
    q1 = q1 / torch.linalg.norm(q1, dim=-1, keepdim=True).clamp_min(1e-8)
    q2 = q2 / torch.linalg.norm(q2, dim=-1, keepdim=True).clamp_min(1e-8)
    dot = torch.abs(torch.sum(q1 * q2, dim=-1)).clamp(max=1.0)
    return 2.0 * torch.arccos(dot)
