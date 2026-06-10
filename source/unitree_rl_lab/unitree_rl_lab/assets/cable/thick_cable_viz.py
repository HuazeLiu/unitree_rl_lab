# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Debug visualization helpers for thick cable validation and RL development."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from .thick_cable_utils import get_cable_centerline_points, get_cable_endpoint_poses, get_cable_segment_poses


def _centerline_marker_cfg(prim_path: str) -> VisualizationMarkersCfg:
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "point": sim_utils.SphereCfg(
                radius=0.003,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.95, 0.25)),
            ),
        },
    )


def _target_marker_cfg(prim_path: str) -> VisualizationMarkersCfg:
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "point": sim_utils.SphereCfg(
                radius=0.010,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
            ),
        },
    )


def _endpoint_frame_cfg(prim_path: str) -> VisualizationMarkersCfg:
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "frame": sim_utils.CylinderCfg(
                radius=0.004,
                height=0.12,
                axis="X",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.75, 1.0)),
            ),
        },
    )


class ThickCableVisualizer:
    """Draw cable centerline, target curve, endpoint frames, and optional curvature colors."""

    def __init__(
        self,
        centerline_path: str = "/Visuals/CableCenterline",
        target_path: str = "/Visuals/CableTarget",
        endpoint_path: str = "/Visuals/CableEndpoints",
    ) -> None:
        self.centerline_markers = VisualizationMarkers(_centerline_marker_cfg(centerline_path))
        self.target_markers = VisualizationMarkers(_target_marker_cfg(target_path))
        self.endpoint_markers = VisualizationMarkers(_endpoint_frame_cfg(endpoint_path))

    def draw_centerline(self, cable: Articulation, env_id: int = 0) -> None:
        points = get_cable_centerline_points(cable)[env_id]
        self.centerline_markers.visualize(translations=points)

    def draw_target_curve(self, target_points: torch.Tensor, env_id: int = 0) -> None:
        if target_points.ndim == 3:
            target_points = target_points[env_id]
        self.target_markers.visualize(translations=target_points)

    def draw_endpoint_frames(self, cable: Articulation, env_id: int = 0) -> None:
        start_pos, start_quat, end_pos, end_quat = get_cable_endpoint_poses(cable)
        positions = torch.stack([start_pos[env_id], end_pos[env_id]], dim=0)
        orientations = torch.stack([start_quat[env_id], end_quat[env_id]], dim=0)
        self.endpoint_markers.visualize(translations=positions, orientations=orientations)

    def draw_segments_colored_by_curvature(self, cable: Articulation, env_id: int = 0) -> None:
        """Color centerline points by local discrete curvature magnitude."""
        centerline = get_cable_centerline_points(cable)[env_id]
        if centerline.shape[0] < 3:
            self.centerline_markers.visualize(translations=centerline)
            return

        second_diff = centerline[2:] - 2.0 * centerline[1:-1] + centerline[:-2]
        curvature = torch.linalg.norm(second_diff, dim=-1)
        curvature = torch.cat([curvature[:1], curvature, curvature[-1:]])
        curvature = curvature / curvature.max().clamp_min(1e-6)

        # Subtle scale variation only (avoid bead-like enlargement).
        scales = torch.ones(curvature.shape[0], 3, device=centerline.device)
        scales[:, 0] = 0.85 + 0.25 * curvature
        scales[:, 1] = scales[:, 0]
        scales[:, 2] = scales[:, 0]
        self.centerline_markers.visualize(translations=centerline, scales=scales)

    def update_all(
        self,
        cable: Articulation,
        target_points: torch.Tensor | None = None,
        env_id: int = 0,
        color_by_curvature: bool = False,
        draw_centerline: bool = True,
        draw_endpoints: bool = True,
    ) -> None:
        if draw_centerline:
            if color_by_curvature:
                self.draw_segments_colored_by_curvature(cable, env_id=env_id)
            else:
                self.draw_centerline(cable, env_id=env_id)
        if target_points is not None:
            self.draw_target_curve(target_points, env_id=env_id)
        if draw_endpoints:
            self.draw_endpoint_frames(cable, env_id=env_id)
