# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""ZED-style egocentric camera mounted on G1 torso (head proxy) for play/deploy."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass


@configclass
class ZedHeadCameraCfg(CameraCfg):
    """Pinhole RGB-D style camera on torso_link (egocentric toward workspace)."""

    # Parent on torso_link (G1 URDF has no zed_link child).
    prim_path: str = "{ENV_REGEX_NS}/Robot/torso_link/ZEDCamera"
    update_period: float = 0.1
    height: int = 360
    width: int = 640
    data_types: list[str] = ["rgb", "distance_to_image_plane"]
    spawn: sim_utils.PinholeCameraCfg = sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=36.0,
        clipping_range=(0.15, 10.0),
    )
    offset: CameraCfg.OffsetCfg = CameraCfg.OffsetCfg(
        pos=(0.08, 0.0, 0.35),
        rot=(0.9848078, 0.0, -0.1736482, 0.0),
        convention="world",
    )


def make_zed_head_camera_cfg(enabled: bool = True, update_period: float = 0.1) -> ZedHeadCameraCfg | None:
    """Return camera cfg for play/deploy, or None when disabled (training)."""
    if not enabled:
        return None
    cfg = ZedHeadCameraCfg()
    cfg.update_period = update_period
    return cfg
