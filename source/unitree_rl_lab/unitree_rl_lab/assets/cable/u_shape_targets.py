# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Parametric U-shape target polylines for cable bending tasks."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vector by quaternion (w, x, y, z). Supports batched quat and vec."""
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)
    if vec.dim() == 1:
        vec = vec.unsqueeze(0)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    uv = torch.stack(
        [
            y * vec[:, 2] - z * vec[:, 1],
            z * vec[:, 0] - x * vec[:, 2],
            x * vec[:, 1] - y * vec[:, 0],
        ],
        dim=-1,
    )
    uuv = torch.stack(
        [
            y * uv[:, 2] - z * uv[:, 1],
            z * uv[:, 0] - x * uv[:, 2],
            x * uv[:, 1] - y * uv[:, 0],
        ],
        dim=-1,
    )
    return vec + 2.0 * (w.unsqueeze(-1) * uv + uuv)


def _quat_inv(quat: torch.Tensor) -> torch.Tensor:
    """Inverse of unit quaternion (w, x, y, z)."""
    inv = quat.clone()
    inv[..., 1:] = -inv[..., 1:]
    return inv


@dataclass(frozen=True)
class UShapePreset:
    """One discrete U-bend depth/span preset."""

    name: str
    u_depth_m: float
    u_span_m: float
    apex_height_offset_m: float = 0.0


# Four training depths (shallow -> deep-narrow).
# Span/depth pairs use circular-arc geometry that is physically achievable by a rubber hose
# pinned at both hands: as hands move closer, the hose naturally forms a circular arc.
# depth_m values are the sag (vertical drop) at the arc apex; span_m is the chord length.
U_SHAPE_DEPTH_PRESETS: tuple[UShapePreset, ...] = (
    # Spans matched to G1 cable-hold hand span: hands at y≈±0.439, cable
    # endpoint segment centers at y≈±0.402.  Half-spans of 0.40/0.375/0.34/0.31
    # keep the target U endpoints close to the actual cable endpoint positions.
    UShapePreset("shallow", u_depth_m=0.06, u_span_m=0.80),
    UShapePreset("med_shallow", u_depth_m=0.10, u_span_m=0.75),
    UShapePreset("med_deep", u_depth_m=0.14, u_span_m=0.68),
    UShapePreset("deep_narrow", u_depth_m=0.16, u_span_m=0.62),
)

NUM_U_DEPTH_PRESETS = len(U_SHAPE_DEPTH_PRESETS)


def generate_u_shape_centerline_local(
    preset: UShapePreset,
    num_points: int = 20,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a symmetric U polyline using circular-arc geometry in the torso-local workspace.

    Uses the same circular arc math as a physical hose pinned at both endpoints.
    Arc lies in the Y-Z plane: Y is left-right (span direction), Z is up-down (sag direction).

    Frame convention: +X forward, +Y left, +Z up.
    The U spans left-right (Y axis) and sags downward (-Z).
    Endpoints sit at ``±span/2`` on Y at z=0; the arc apex dips to ``-depth``.

    Returns:
        Tensor of shape ``(num_points, 3)``.
    """
    import math

    if num_points < 3:
        raise ValueError("num_points must be >= 3")

    span = preset.u_span_m
    depth = preset.u_depth_m
    half_span = 0.5 * span

    if depth < 1e-6:
        y = torch.linspace(-half_span, half_span, num_points, device=device, dtype=dtype)
        z = torch.zeros(num_points, device=device, dtype=dtype)
    else:
        radius = (span ** 2) / (8.0 * depth) + depth / 2.0
        half_angle = math.asin(min(half_span / radius, 0.9999))
        t = torch.linspace(-half_angle, half_angle, num_points, device=device, dtype=dtype)
        y = radius * torch.sin(t)
        z = radius * (torch.cos(t) - 1.0) + depth

    x = torch.zeros_like(y)
    return torch.stack([x, y, z], dim=-1)


def generate_u_shape_centerline_batch(
    depth_ids: torch.Tensor,
    torso_pos_w: torch.Tensor,
    torso_quat_w: torch.Tensor,
    workspace_offset_body: tuple[float, float, float] = (0.35, 0.0, 0.12),
    num_points: int = 20,
) -> torch.Tensor:
    """Generate world-frame U targets for a batch of environments.

    Args:
        depth_ids: Integer preset indices, shape ``(num_envs,)``.
        torso_pos_w: Torso positions in world frame, shape ``(num_envs, 3)``.
        torso_quat_w: Torso orientations (w, x, y, z), shape ``(num_envs, 4)``.
        workspace_offset_body: Origin of the U workspace in torso frame.

    Returns:
        Target centerline points, shape ``(num_envs, num_points, 3)``.
    """
    num_envs = depth_ids.shape[0]
    device = depth_ids.device
    dtype = torso_pos_w.dtype
    offset_vec = torch.tensor(workspace_offset_body, device=device, dtype=dtype)

    targets = torch.zeros(num_envs, num_points, 3, device=device, dtype=dtype)
    for env_i in range(num_envs):
        preset_id = int(depth_ids[env_i].item())
        local_curve = generate_u_shape_centerline_local(
            U_SHAPE_DEPTH_PRESETS[preset_id], num_points=num_points, device=device, dtype=dtype
        )
        pos = torso_pos_w[env_i] + _quat_apply(torso_quat_w[env_i], offset_vec).squeeze(0)
        quat = torso_quat_w[env_i]
        rotated = _quat_apply(
            quat.unsqueeze(0).expand(num_points, -1),
            local_curve,
        )
        targets[env_i] = pos.unsqueeze(0) + rotated

    return targets


def get_u_shape_endpoint_targets(
    target_centerline: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract start/end positions and default orientations from a target polyline.

    Returns:
        start_pos, start_quat, end_pos, end_quat with shapes ``(N, 3)`` and ``(N, 4)``.
    """
    start_pos = target_centerline[:, 0]
    end_pos = target_centerline[:, -1]
    batch = target_centerline.shape[0]
    device = target_centerline.device
    dtype = target_centerline.dtype
    identity = torch.zeros(batch, 4, device=device, dtype=dtype)
    identity[:, 0] = 1.0
    return start_pos, identity, end_pos, identity


def depth_id_one_hot(depth_ids: torch.Tensor, num_presets: int = NUM_U_DEPTH_PRESETS) -> torch.Tensor:
    """One-hot encoding of depth indices, shape ``(num_envs, num_presets)``."""
    out = torch.zeros(depth_ids.shape[0], num_presets, device=depth_ids.device, dtype=torch.float32)
    out.scatter_(1, depth_ids.long().unsqueeze(1), 1.0)
    return out


def normalized_depth(depth_ids: torch.Tensor, num_presets: int = NUM_U_DEPTH_PRESETS) -> torch.Tensor:
    """Normalized depth id in [0, 1], shape ``(num_envs, 1)``."""
    if num_presets <= 1:
        return torch.zeros(depth_ids.shape[0], 1, device=depth_ids.device)
    return (depth_ids.float() / float(num_presets - 1)).unsqueeze(1)
