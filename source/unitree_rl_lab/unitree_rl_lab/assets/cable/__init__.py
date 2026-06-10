"""Thick cable assets — articulated rigid-body rope approximation for RL."""

from .thick_cable_builder import ThickCableBuildResult, create_thick_cable
from .thick_cable_cfg import (
    CABLE_LIKE_CFG,
    SOFT_CABLE_CFG,
    STIFF_FLAT_TUBE_CFG,
    THICK_CABLE_DEFAULT_CFG,
    ThickCableCfg,
    make_thick_cable_articulation_cfg,
)
from .thick_cable_utils import (
    compute_bending_energy,
    compute_centerline_chamfer_reward,
    compute_endpoint_reward,
    compute_smoothness_reward,
    get_cable_centerline_points,
    get_cable_endpoint_poses,
    get_cable_segment_poses,
)
from .thick_cable_viz import ThickCableVisualizer

__all__ = [
    "ThickCableBuildResult",
    "ThickCableCfg",
    "ThickCableVisualizer",
    "THICK_CABLE_DEFAULT_CFG",
    "STIFF_FLAT_TUBE_CFG",
    "SOFT_CABLE_CFG",
    "CABLE_LIKE_CFG",          # alias for SOFT_CABLE_CFG
    "create_thick_cable",
    "make_thick_cable_articulation_cfg",
    "get_cable_segment_poses",
    "get_cable_centerline_points",
    "get_cable_endpoint_poses",
    "compute_centerline_chamfer_reward",
    "compute_endpoint_reward",
    "compute_bending_energy",
    "compute_smoothness_reward",
]
