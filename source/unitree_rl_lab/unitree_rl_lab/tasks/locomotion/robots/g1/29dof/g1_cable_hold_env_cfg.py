# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""G1 standing still with arms raised, holding a thick cable at both hands."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.cable.thick_cable_cfg import CABLE_LIKE_CFG, make_thick_cable_articulation_cfg
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG
from unitree_rl_lab.tasks.locomotion import mdp

from . import velocity_armhold_env_cfg as armhold
from . import velocity_env_cfg as base_cfg

# Arms lifted forward/outward so hands span ~cable length (0.8 m) at chest height.
CABLE_HOLD_ARM_POSE: dict[str, float] = armhold._full_arm_pose(
    left_shoulder_pitch_joint=-0.72,
    right_shoulder_pitch_joint=-0.72,
    left_shoulder_roll_joint=0.92,
    right_shoulder_roll_joint=-0.92,
    left_shoulder_yaw_joint=0.12,
    right_shoulder_yaw_joint=-0.12,
    left_elbow_joint=1.35,
    right_elbow_joint=1.35,
    left_wrist_roll_joint=0.0,
    right_wrist_roll_joint=0.0,
    left_wrist_pitch_joint=-0.12,
    right_wrist_pitch_joint=-0.12,
    left_wrist_yaw_joint=0.0,
    right_wrist_yaw_joint=0.0,
)

STANDING_LOWER_BODY_POSE: dict[str, float] = {
    "left_hip_pitch_joint": -0.1,
    "right_hip_pitch_joint": -0.1,
    "left_hip_roll_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.3,
    "right_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "right_ankle_pitch_joint": -0.2,
    "left_ankle_roll_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
}

G1_CABLE_HOLD_POSE: dict[str, float] = {**STANDING_LOWER_BODY_POSE, **CABLE_HOLD_ARM_POSE}

G1_CABLE_SCENE_CABLE_CFG = CABLE_LIKE_CFG.replace(prim_path="{ENV_REGEX_NS}/Cable")


def _build_g1_cable_hold_robot_cfg() -> ArticulationCfg:
    base_joint_pos = dict(UNITREE_G1_29DOF_CFG.init_state.joint_pos)
    for key in list(base_joint_pos.keys()):
        if any(token in key for token in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "waist")):
            del base_joint_pos[key]
    base_joint_pos.update(G1_CABLE_HOLD_POSE)
    return UNITREE_G1_29DOF_CFG.replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=UNITREE_G1_29DOF_CFG.init_state.pos,
            joint_pos=base_joint_pos,
            joint_vel=UNITREE_G1_29DOF_CFG.init_state.joint_vel,
        )
    )


@configclass
class G1CableHoldSceneCfg(base_cfg.RobotSceneCfg):
    """G1 robot plus thick cable articulation."""

    robot: ArticulationCfg = _build_g1_cable_hold_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
    cable: ArticulationCfg = make_thick_cable_articulation_cfg(G1_CABLE_SCENE_CABLE_CFG)


@configclass
class G1CableHoldEventCfg:
    """Reset events for a static cable-hold vignette (no locomotion randomization)."""

    hold_pose_reset = EventTerm(
        func=mdp.hold_joint_position_targets,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_positions": G1_CABLE_HOLD_POSE,
            "write_joint_state": True,
        },
    )
    attach_cable = EventTerm(
        func=mdp.attach_g1_cable_to_hands,
        mode="reset",
    )
    hold_pose_interval = EventTerm(
        func=mdp.hold_joint_position_targets,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_positions": G1_CABLE_HOLD_POSE,
            "write_joint_state": False,
        },
    )


@configclass
class G1CableHoldActionsCfg(base_cfg.ActionsCfg):
    """Zero locomotion commands; arms/legs held via events."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=armhold.LOCOMOTION_JOINT_NAMES,
        scale=0.0,
        use_default_offset=True,
    )
    ArmHoldPositionAction = mdp.ArmHoldPositionActionCfg(
        asset_name="robot",
        joint_names=armhold.ARM_JOINT_NAMES,
        target_positions=CABLE_HOLD_ARM_POSE,
    )


@configclass
class G1CableHoldEnvCfg(base_cfg.RobotEnvCfg):
    """Train/debug cfg (single env spacing for inspection)."""

    scene: G1CableHoldSceneCfg = G1CableHoldSceneCfg(num_envs=1, env_spacing=4.0)
    actions: G1CableHoldActionsCfg = G1CableHoldActionsCfg()
    events: G1CableHoldEventCfg = G1CableHoldEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.sim.dt = G1_CABLE_SCENE_CABLE_CFG.physics_dt
        self.decimation = max(1, int(round((1.0 / 30.0) / self.sim.dt)))


@configclass
class G1CableHoldPlayEnvCfg(G1CableHoldEnvCfg):
    """GUI / video: flat ground, fixed root, cable welded to hands."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        self.observations.policy.enable_corruption = False
        self.curriculum.terrain_levels = None
        self.commands.base_velocity.debug_vis = False
        self.rerender_on_reset = True
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (2.2, -2.0, 1.45)
        self.viewer.lookat = (0.35, 0.0, 0.95)
