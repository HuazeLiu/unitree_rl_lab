# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the articulated rigid-body cable model.

A cable is modeled as a chain of capsule links connected by passive D6 joints
with configurable stiffness, damping, and joint limits.  Two presets are provided:

``STIFF_FLAT_TUBE_CFG`` — Production preset for RL manipulation tasks.
    Stiff enough to stay nearly straight under gravity (< 2 cm sag over 0.88 m),
    bendable enough for G1 humanoid arms.  20 segments, solver 32/8 at 240 Hz.

``SOFT_CABLE_CFG`` — Soft preset for visualisation / debugging.
    Lower stiffness allows natural sag; used by the cable-hold vignette.
"""

from __future__ import annotations

from collections.abc import Callable

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from .thick_cable_builder import create_thick_cable


# ---------------------------------------------------------------------------
# Spawner configuration
# ---------------------------------------------------------------------------


@configclass
class ThickCableSpawnerCfg(sim_utils.SpawnerCfg):
    """Spawner that builds a thick cable articulation procedurally at runtime."""

    func: Callable = create_thick_cable

    # Geometry
    total_length: float = 0.80
    radius: float = 0.025
    num_segments: int = 20

    # Inertial
    total_mass: float = 1.5

    # Joint stiffness / damping  (N·m/rad, N·m·s/rad)
    bend_stiffness: float = 5.0
    bend_damping: float = 0.4
    twist_stiffness: float = 2.0
    twist_damping: float = 0.15

    # Joint limits (degrees)
    bend_limit_deg: float = 6.0
    twist_limit_deg: float = 2.0

    # Actuator caps
    max_joint_effort: float = 8.0
    max_joint_velocity: float = 20.0

    # Contact
    static_friction: float = 1.0
    dynamic_friction: float = 0.9
    restitution: float = 0.0
    contact_offset: float = 0.003
    rest_offset: float = 0.0
    self_collision: bool = False
    collision_enabled: bool = True

    # Physics
    disable_gravity: bool = False
    kinematic_enabled: bool = False
    position_solver_iterations: int = 32
    velocity_solver_iterations: int = 8


# ---------------------------------------------------------------------------
# Cable configuration dataclass
# ---------------------------------------------------------------------------


@configclass
class ThickCableCfg:
    """Full cable configuration including simulation, geometry, and spawn parameters.

    Use ``.replace(…)`` to override fields for a specific scene.
    """

    prim_path: str = "/World/Cable"

    # Geometry
    total_length: float = 0.80
    radius: float = 0.025
    num_segments: int = 20

    # Inertial
    total_mass: float = 1.5

    # Joint properties
    bend_stiffness: float = 5.0
    bend_damping: float = 0.4
    twist_stiffness: float = 2.0
    twist_damping: float = 0.15
    bend_limit_deg: float = 6.0
    twist_limit_deg: float = 2.0
    max_joint_effort: float = 8.0
    max_joint_velocity: float = 20.0

    # Contact
    static_friction: float = 1.0
    dynamic_friction: float = 0.9
    restitution: float = 0.0
    contact_offset: float = 0.003
    rest_offset: float = 0.0
    self_collision: bool = False
    collision_enabled: bool = True

    # Physics
    disable_gravity: bool = False
    kinematic_enabled: bool = False
    physics_dt: float = 1.0 / 240.0
    position_solver_iterations: int = 32
    velocity_solver_iterations: int = 8

    # Initial pose (world frame)
    init_pos: tuple[float, float, float] = (0.0, 0.0, 1.0)
    init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # ---- derived ----
    @property
    def segment_length(self) -> float:
        return self.total_length / self.num_segments

    @property
    def segment_mass(self) -> float:
        return self.total_mass / self.num_segments

    def to_spawner_cfg(self) -> ThickCableSpawnerCfg:
        return ThickCableSpawnerCfg(
            total_length=self.total_length,
            radius=self.radius,
            num_segments=self.num_segments,
            total_mass=self.total_mass,
            bend_stiffness=self.bend_stiffness,
            bend_damping=self.bend_damping,
            twist_stiffness=self.twist_stiffness,
            twist_damping=self.twist_damping,
            bend_limit_deg=self.bend_limit_deg,
            twist_limit_deg=self.twist_limit_deg,
            max_joint_effort=self.max_joint_effort,
            max_joint_velocity=self.max_joint_velocity,
            static_friction=self.static_friction,
            dynamic_friction=self.dynamic_friction,
            restitution=self.restitution,
            contact_offset=self.contact_offset,
            rest_offset=self.rest_offset,
            self_collision=self.self_collision,
            position_solver_iterations=self.position_solver_iterations,
            velocity_solver_iterations=self.velocity_solver_iterations,
            disable_gravity=self.disable_gravity,
            kinematic_enabled=self.kinematic_enabled,
            collision_enabled=self.collision_enabled,
        )


# ---------------------------------------------------------------------------
# Named presets
# ---------------------------------------------------------------------------

# Production preset for RL manipulation. 20 segments, 5.0 N·m/rad bend
# stiffness — stiff enough for < 2 cm sag under gravity, bendable enough
# for G1 arms (≈ 32 N per hand for the deepest U preset).  6° per-joint
# limit supports all four U-depth presets (shallow → deep-narrow).
STIFF_FLAT_TUBE_CFG = ThickCableCfg(
    num_segments=20,
    total_mass=1.5,
    bend_stiffness=5.0,
    bend_damping=0.4,
    twist_stiffness=2.0,
    twist_damping=0.15,
    bend_limit_deg=6.0,
    twist_limit_deg=2.0,
    max_joint_effort=8.0,
    max_joint_velocity=20.0,
    static_friction=1.0,
    dynamic_friction=0.9,
    collision_enabled=True,
    self_collision=False,
    disable_gravity=False,
    position_solver_iterations=32,
    velocity_solver_iterations=8,
    physics_dt=1.0 / 240.0,
)

# Soft preset for visualisation / debugging.  24 segments, low stiffness
# allows natural gravity sag.  Used by the cable-hold vignette.
SOFT_CABLE_CFG = ThickCableCfg(
    num_segments=24,
    total_mass=0.95,
    bend_stiffness=0.32,
    bend_damping=0.028,
    twist_stiffness=0.045,
    twist_damping=0.014,
    bend_limit_deg=4.5,
    twist_limit_deg=1.2,
    max_joint_effort=3.0,
    max_joint_velocity=12.0,
    position_solver_iterations=24,
    velocity_solver_iterations=8,
    physics_dt=1.0 / 240.0,
)

# Backward-compatibility alias — the soft preset was historically called
# "cable-like".  Kept for the cable-hold task and attachment fallbacks.
CABLE_LIKE_CFG = SOFT_CABLE_CFG

# Default for make_thick_cable_articulation_cfg when no preset is given.
THICK_CABLE_DEFAULT_CFG = STIFF_FLAT_TUBE_CFG


# ---------------------------------------------------------------------------
# Articulation factory
# ---------------------------------------------------------------------------

# PhysX D6 joint DOF naming:  :0 = twist (rotX),  :1, :2 = bend (rotY, rotZ)
_CABLE_TWIST_DOF_EXPR = [r".*:0"]
_CABLE_BEND_DOF_EXPR = [r".*:1", r".*:2"]


def make_thick_cable_articulation_cfg(cfg: ThickCableCfg | None = None) -> ArticulationCfg:
    """Create an Isaac Lab ``ArticulationCfg`` for a thick cable.

    Bend and twist axes receive separate ``ImplicitActuatorCfg`` entries so
    PhysX applies the correct per-DOF stiffness/damping.
    """
    cable_cfg = cfg or THICK_CABLE_DEFAULT_CFG
    return ArticulationCfg(
        prim_path=cable_cfg.prim_path,
        spawn=cable_cfg.to_spawner_cfg(),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=cable_cfg.init_pos,
            rot=cable_cfg.init_rot,
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "bend": ImplicitActuatorCfg(
                joint_names_expr=_CABLE_BEND_DOF_EXPR,
                stiffness=cable_cfg.bend_stiffness,
                damping=cable_cfg.bend_damping,
                effort_limit_sim=cable_cfg.max_joint_effort,
                velocity_limit_sim=cable_cfg.max_joint_velocity,
            ),
            "twist": ImplicitActuatorCfg(
                joint_names_expr=_CABLE_TWIST_DOF_EXPR,
                stiffness=cable_cfg.twist_stiffness,
                damping=cable_cfg.twist_damping,
                effort_limit_sim=cable_cfg.max_joint_effort,
                velocity_limit_sim=cable_cfg.max_joint_velocity,
            ),
        },
    )
