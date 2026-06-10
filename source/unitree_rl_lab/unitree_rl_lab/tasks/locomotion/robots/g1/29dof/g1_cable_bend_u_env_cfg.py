# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""G1 whole-body cable U-bend manipulation — production environment config.

A 20-segment stiff articulated cable is pinned to the G1's hands via GPU-safe
point welds.  The policy controls 17 DOF (arms + waist) to bend the cable into
a U-shape matching a commanded depth preset.  Training runs at 8192 environments
on a single RTX 4090 (24 GB).
"""

from __future__ import annotations

import math
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.cable.thick_cable_cfg import (
    STIFF_FLAT_TUBE_CFG,
    make_thick_cable_articulation_cfg,
)
from unitree_rl_lab.assets.cable.zed_head_camera_cfg import make_zed_head_camera_cfg
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp.cable.commands import CableBendUCommandCfg

from . import g1_cable_hold_env_cfg as cable_hold


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TABLE_HEIGHT = 0.75

# Hand positions from G1 URDF at the cable-hold pose.
# Left  ≈ (0.263,  0.439, 0.978)   Right ≈ (0.263, -0.439, 0.978)
# Hand-span = 0.878 m  →  cable total_length = 0.878 m (taut at rest).
_HAND_SPAN = 0.878

# 20 segments → half_seg = 0.878 / 40 = 0.02195
# seg_00 centre Y = 0.439 + dir_n * half_seg  (dir_n = (0, -1, 0))  →  0.417
_CABLE_INIT_Y = 0.417

CurriculumStage = Literal["bend", "full"]


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

G1_CABLE_BEND_SCENE_CABLE_CFG = STIFF_FLAT_TUBE_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Cable",
    init_pos=(0.263, _CABLE_INIT_Y, 0.978),
    init_rot=(0.70710678, 0.0, 0.0, -0.70710678),
    total_length=_HAND_SPAN,
)


def _build_g1_cable_bend_robot_cfg() -> ArticulationCfg:
    """G1 robot starting at the cable-hold arm pose."""
    base_joint_pos = dict(UNITREE_G1_29DOF_CFG.init_state.joint_pos)
    for key in list(base_joint_pos.keys()):
        if any(t in key for t in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "waist")):
            del base_joint_pos[key]
    base_joint_pos.update(cable_hold.G1_CABLE_HOLD_POSE)
    return UNITREE_G1_29DOF_CFG.replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=UNITREE_G1_29DOF_CFG.init_state.pos,
            joint_pos=base_joint_pos,
            joint_vel=UNITREE_G1_29DOF_CFG.init_state.joint_vel,
        )
    )


@configclass
class G1CableBendUSceneCfg(InteractiveSceneCfg):
    """G1 robot, stiff cable, table, contact sensors.  ZED camera optional (play only)."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = _build_g1_cable_bend_robot_cfg().replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    cable: ArticulationCfg = make_thick_cable_articulation_cfg(
        G1_CABLE_BEND_SCENE_CABLE_CFG
    )
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.5, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.55, 0.58), roughness=0.55
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2, 0.0, TABLE_HEIGHT)),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )


# ---------------------------------------------------------------------------
# Actions — 17 DOF (14 arms + 3 waist)
# ---------------------------------------------------------------------------

CABLE_BEND_ARM_JOINTS = [
    "left_shoulder_pitch_joint",  "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",   "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",    "right_shoulder_yaw_joint",
    "left_elbow_joint",           "right_elbow_joint",
    "left_wrist_roll_joint",      "right_wrist_roll_joint",
    "left_wrist_pitch_joint",     "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",       "right_wrist_yaw_joint",
]
CABLE_BEND_WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
CABLE_BEND_POLICY_JOINTS = CABLE_BEND_ARM_JOINTS + CABLE_BEND_WAIST_JOINTS

# Leg joints locked at standing pose via PD control (fix_root_link = True,
# so the robot cannot fall; kinematic lock is unnecessary).
CABLE_BEND_LEG_POSE = {
    name: val
    for name, val in cable_hold.STANDING_LOWER_BODY_POSE.items()
    if "waist" not in name
}
CABLE_BEND_LEG_JOINTS = list(CABLE_BEND_LEG_POSE.keys())


@configclass
class G1CableBendUActionsCfg:
    """17-DOF policy + locked legs."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=CABLE_BEND_POLICY_JOINTS,
        scale=0.02,
        use_default_offset=True,
    )

    LegHoldPositionAction = mdp.ArmHoldPositionActionCfg(
        asset_name="robot",
        joint_names=CABLE_BEND_LEG_JOINTS,
        target_positions=CABLE_BEND_LEG_POSE,
        kinematic_lock=False,
    )


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

@configclass
class G1CableBendUObservationsCfg:
    """Policy and critic observation groups."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_policy = ObsTerm(
            func=mdp.joint_pos_policy, noise=Unoise(n_min=-0.005, n_max=0.005)
        )
        joint_vel_policy = ObsTerm(
            func=mdp.joint_vel_policy, scale=0.05, noise=Unoise(n_min=-1.0, n_max=1.0)
        )
        last_action = ObsTerm(func=mdp.last_action)
        cable_bend_command = ObsTerm(
            func=mdp.cable_bend_command,
            params={"command_name": "cable_bend_u"},
            clip=(-2.0, 2.0),
        )
        cable_endpoints_body = ObsTerm(
            func=mdp.cable_endpoints_body, clip=(-5.0, 5.0)
        )
        cable_centerline_sparse = ObsTerm(
            func=mdp.cable_centerline_sparse,
            params={"num_points": 8},
            clip=(-5.0, 5.0),
        )
        target_centerline = ObsTerm(
            func=mdp.target_centerline_flat,
            params={"command_name": "cable_bend_u"},
            clip=(-5.0, 5.0),
        )
        hand_pos_body = ObsTerm(func=mdp.hand_pos_body, clip=(-5.0, 5.0))
        shape_error = ObsTerm(func=mdp.cable_shape_error, clip=(0.0, 1.0))
        phase_progress = ObsTerm(func=mdp.phase_progress, clip=(0.0, 1.0))

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_policy = ObsTerm(func=mdp.joint_pos_policy)
        joint_vel_policy = ObsTerm(func=mdp.joint_vel_policy, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        cable_bend_command = ObsTerm(
            func=mdp.cable_bend_command,
            params={"command_name": "cable_bend_u"},
            clip=(-2.0, 2.0),
        )
        cable_endpoints_body = ObsTerm(func=mdp.cable_endpoints_body, clip=(-5.0, 5.0))
        cable_centerline_sparse = ObsTerm(
            func=mdp.cable_centerline_sparse,
            params={"num_points": 8},
            clip=(-5.0, 5.0),
        )
        hand_pos_body = ObsTerm(func=mdp.hand_pos_body)
        hand_contact_force = ObsTerm(
            func=mdp.hand_contact_force,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*rubber_hand"])
            },
        )
        cable_centerline_full_body = ObsTerm(func=mdp.cable_centerline_full_body)
        target_centerline_flat = ObsTerm(func=mdp.target_centerline_flat)
        phase_progress = ObsTerm(func=mdp.phase_progress, clip=(0.0, 1.0))

        def __post_init__(self):
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

@configclass
class G1CableBendURewardsCfg:
    """Reward terms for the U-bend task.

    The shape-matching term dominates (30× weight, temperature = 0.10).
    Symmetry reward (10×) encourages mirrored arm motion, effectively halving
    the search space.  Depth and span terms are secondary.  Penalties are kept
    10-50× smaller than the shaping terms so they nudge rather than dominate.
    """

    # ---- Primary shaping ----
    bend_shape = RewTerm(
        func=mdp.bend_shape_reward,
        weight=30.0,
        params={"command_name": "cable_bend_u", "temperature": 0.10},
    )
    bend_endpoints = RewTerm(
        func=mdp.bend_endpoints_reward,
        weight=3.0,
        params={"command_name": "cable_bend_u", "temperature": 0.10},
    )
    bend_depth = RewTerm(
        func=mdp.bend_depth_reward,
        weight=1.0,
        params={"command_name": "cable_bend_u", "temperature": 0.03},
    )
    bend_smooth = RewTerm(
        func=mdp.bend_smooth_reward, weight=0.5, params={"temperature": 0.06}
    )

    # ---- Bilateral symmetry (halves effective search space) ----
    arm_symmetry = RewTerm(func=mdp.arm_symmetry_reward, weight=10.0)

    # ---- Progress, settling, contact ----
    bend_stop = RewTerm(func=mdp.bend_stop_reward, weight=0.5)
    bend_settle = RewTerm(func=mdp.bend_settle_reward, weight=5.0)
    bend_contact = RewTerm(func=mdp.bend_contact_stability, weight=0.3)
    bend_progress = RewTerm(
        func=mdp.bend_progress_reward,
        weight=5.0,
        params={"command_name": "cable_bend_u", "normalization": 0.0001},
    )

    # ---- Auxiliary ----
    alive = RewTerm(func=mdp.is_alive, weight=0.20)
    hand_proximity = RewTerm(
        func=mdp.hand_proximity_reward,
        weight=1.5,
        params={"command_name": "cable_bend_u"},
    )

    # ---- Penalties (kept small — nudge, don't dominate) ----
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01)
    joint_vel = RewTerm(func=mdp.policy_joint_vel_l2, weight=-0.00005)
    joint_acc = RewTerm(func=mdp.policy_joint_acc_l2, weight=-1.0e-8)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_pos_limits = RewTerm(func=mdp.policy_dof_pos_limits, weight=-0.03)
    energy = RewTerm(func=mdp.policy_energy, weight=-1e-6)

    # ---- Stage-1 (zeroed in bend-only curriculum) ----
    reach_left = RewTerm(
        func=mdp.reach_hand_to_cable_end,
        weight=0.0,
        params={"side": "left", "temperature": 0.05},
    )
    reach_right = RewTerm(
        func=mdp.reach_hand_to_cable_end,
        weight=0.0,
        params={"side": "right", "temperature": 0.05},
    )
    lift_cable = RewTerm(func=mdp.lift_cable_reward, weight=0.0)
    dual_grasp = RewTerm(
        func=mdp.dual_grasp_reward, weight=0.0, params={"grasp_threshold": 0.08}
    )
    lower_body_dev = RewTerm(func=mdp.lower_body_deviation, weight=0.0)


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------

@configclass
class G1CableBendUTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cable_invalid = DoneTerm(func=mdp.cable_state_invalid)
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.25}
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation, params={"limit_angle": 0.85}
    )
    bend_success = DoneTerm(
        func=mdp.cable_bend_success,
        params={"success_threshold": 0.85, "hold_steps": 20},
    )


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@configclass
class G1CableBendUEventCfg:
    """Domain randomisation and reset events."""

    # Startup (once)
    setup_state = EventTerm(
        func=mdp.setup_cable_bend_state,
        mode="startup",
        params={"curriculum_stage": "bend", "table_height": TABLE_HEIGHT},
    )
    clear_attach_cache = EventTerm(
        func=mdp.clear_attachment_cache_on_startup, mode="startup"
    )
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-2.0, 2.0),
            "operation": "add",
        },
    )
    cable_physics_rand = EventTerm(
        func=mdp.randomize_cable_physics, mode="startup"
    )
    weld_cable_buildtime = EventTerm(
        func=mdp.weld_cable_on_prestartup,
        mode="startup",
        params={"cable_hold_cfg": G1_CABLE_BEND_SCENE_CABLE_CFG},
    )

    # Reset (per episode)
    reset_episode_state = EventTerm(
        func=mdp.reset_cable_bend_episode_state, mode="reset"
    )
    reset_hold_pose = EventTerm(
        func=mdp.hold_joint_position_targets,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_positions": cable_hold.G1_CABLE_HOLD_POSE,
            "write_joint_state": True,
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.97, 1.03), "velocity_range": (0.0, 0.0)},
    )
    attach_cable_hands = EventTerm(
        func=mdp.attach_g1_cable_on_reset,
        mode="reset",
        params={"cable_hold_cfg": G1_CABLE_BEND_SCENE_CABLE_CFG},
    )

    # Interval (every step)
    dual_grasp_attach = EventTerm(
        func=mdp.try_attach_on_dual_grasp,
        mode="interval",
        interval_range_s=(0.0, 0.0),
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@configclass
class G1CableBendUEnvCfg(ManagerBasedRLEnvCfg):
    """Production training config for G1 cable U-bend manipulation.

    8192 environments, 20-segment stiff cable, 17-DOF policy (arms + waist).
    3000 iterations × 32 steps/env = 78.6M total timesteps.
    """

    scene: G1CableBendUSceneCfg = G1CableBendUSceneCfg(num_envs=8192, env_spacing=2.5)
    observations: G1CableBendUObservationsCfg = G1CableBendUObservationsCfg()
    actions: G1CableBendUActionsCfg = G1CableBendUActionsCfg()
    commands: G1CableBendUCommandsCfg = G1CableBendUCommandsCfg()
    rewards: G1CableBendURewardsCfg = G1CableBendURewardsCfg()
    terminations: G1CableBendUTerminationsCfg = G1CableBendUTerminationsCfg()
    events: G1CableBendUEventCfg = G1CableBendUEventCfg()
    curriculum: object = configclass(object)()

    # Runtime overrides
    curriculum_stage: CurriculumStage = "bend"
    table_height: float = TABLE_HEIGHT
    enable_zed_camera: bool = False

    def __post_init__(self):
        # Physics
        self.sim.dt = G1_CABLE_BEND_SCENE_CABLE_CFG.physics_dt          # 1/240 s
        self.decimation = max(1, int(round((1.0 / 30.0) / self.sim.dt)))  # 8
        self.sim.render_interval = self.decimation
        self.episode_length_s = 15.0
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # Fix root link — the robot stands still, arms do the work
        self.scene.robot.spawn.articulation_props.fix_root_link = True

        # Bend-only curriculum: disable stage-1 (pick/place) events & rewards
        self.events.setup_state.params["curriculum_stage"] = self.curriculum_stage
        self.events.setup_state.params["table_height"] = self.table_height
        self.events.reset_robot_joints = EventTerm(func=mdp.event_noop, mode="reset")
        self.events.reset_table = EventTerm(func=mdp.event_noop, mode="reset")

        # Contact sensor update at physics rate
        self.scene.contact_forces.update_period = self.sim.dt


# ---------------------------------------------------------------------------
# Play / evaluation config
# ---------------------------------------------------------------------------

@configclass
class G1CableBendUPlayEnvCfg(G1CableBendUEnvCfg):
    """GUI play: 4 environments, debug visualisation, optional ZED camera."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.observations.policy.enable_corruption = False
        self.commands.cable_bend_u.debug_vis = True

        zed = make_zed_head_camera_cfg(enabled=self.enable_zed_camera)
        if zed is not None:
            self.scene.zed_camera = zed

        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (2.2, -2.0, 1.45)
        self.viewer.lookat = (0.35, 0.0, 0.95)
        self.rerender_on_reset = True
