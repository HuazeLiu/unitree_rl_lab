# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Convert a 2D cable mask (+ optional depth) into sparse centerline points."""

from __future__ import annotations

import numpy as np
import torch


def mask_to_centerline_2d(mask: np.ndarray, num_points: int = 8) -> np.ndarray:
    """Extract ordered 2D centerline points from a binary mask.

    Uses PCA major axis projection and uniform sampling along arclength.

    Returns:
        Array of shape ``(num_points, 2)`` in pixel coordinates (u, v).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 4:
        h, w = mask.shape
        return np.stack(
            [np.linspace(0.2 * w, 0.8 * w, num_points), np.full(num_points, 0.6 * h)],
            axis=-1,
        )

    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=-1)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = centered.T @ centered / max(len(pts), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    proj = centered @ axis
    order = np.argsort(proj)
    ordered = pts[order]

    if len(ordered) < num_points:
        indices = np.linspace(0, len(ordered) - 1, num_points).astype(int)
    else:
        indices = np.linspace(0, len(ordered) - 1, num_points).astype(int)
    return ordered[indices]


def project_centerline_to_body_frame(
    pixels_uv: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
    cam_pos_w: torch.Tensor,
    cam_quat_w: torch.Tensor,
    torso_pos_w: torch.Tensor,
    torso_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Back-project pixels to 3D and express in torso frame.

    Returns:
        Tensor of shape ``(num_points, 3)``.
    """
    import isaaclab.utils.math as math_utils

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    points_w = []
    for u, v in pixels_uv:
        ui, vi = int(np.clip(u, 0, depth_map.shape[1] - 1)), int(np.clip(v, 0, depth_map.shape[0] - 1))
        z = float(depth_map[vi, ui])
        if z <= 0.0 or not np.isfinite(z):
            z = 1.0
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        cam_pt = torch.tensor([x, y, z], device=cam_pos_w.device, dtype=cam_pos_w.dtype)
        world = cam_pos_w + math_utils.quat_apply(cam_quat_w, cam_pt)
        rel = world - torso_pos_w
        body = math_utils.quat_apply(math_utils.quat_inv(torso_quat_w), rel)
        points_w.append(body)
    return torch.stack(points_w, dim=0)
