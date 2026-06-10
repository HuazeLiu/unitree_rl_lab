import math

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from . import velocity_env_cfg as base_cfg

LOCOMOTION_JOINT_NAMES = [
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]

ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]


def _full_arm_pose(**overrides: float) -> dict[str, float]:
    pose = {name: 0.0 for name in ARM_JOINT_NAMES}
    pose.update(overrides)
    return pose


# Pose presets for controlled experiments (everything else identical).
POSE_PRESETS: dict[str, dict[str, float]] = {
    # Guard / boxing ready (URDF g1_29dof_rev_1_0: shoulder_pitch axis +Y, negative = arm forward).
    # Pinocchio FK tuned for fists ~0.30 m ahead of torso, elbows ~90 deg, mid-chest height.
    "guard": _full_arm_pose(
        left_shoulder_pitch_joint=-1.20,
        right_shoulder_pitch_joint=-1.20,
        left_shoulder_roll_joint=0.20,
        right_shoulder_roll_joint=-0.20,
        left_shoulder_yaw_joint=0.0,
        right_shoulder_yaw_joint=0.0,
        left_elbow_joint=1.50,
        right_elbow_joint=1.50,
        left_wrist_roll_joint=0.0,
        right_wrist_roll_joint=0.0,
        left_wrist_pitch_joint=-0.30,
        right_wrist_pitch_joint=-0.30,
        left_wrist_yaw_joint=0.0,
        right_wrist_yaw_joint=0.0,
    ),
    "low": _full_arm_pose(
        left_shoulder_pitch_joint=0.3,
        right_shoulder_pitch_joint=0.3,
        left_shoulder_roll_joint=0.25,
        right_shoulder_roll_joint=-0.25,
        left_elbow_joint=0.97,
        right_elbow_joint=0.97,
        left_wrist_roll_joint=0.15,
        right_wrist_roll_joint=-0.15,
    ),
    "forward": _full_arm_pose(
        left_shoulder_pitch_joint=math.pi / 2,
        right_shoulder_pitch_joint=math.pi / 2,
        left_shoulder_roll_joint=0.25,
        right_shoulder_roll_joint=-0.25,
    ),
    "tpose": _full_arm_pose(
        left_shoulder_roll_joint=math.pi / 2,
        right_shoulder_roll_joint=-math.pi / 2,
    ),
}

# Default for Unitree-G1-29dof-Velocity-ArmHold (guard stance, not T-pose).
ARM_HOLD_POSE = POSE_PRESETS["guard"]


def _build_robot_cfg(arm_hold_pose: dict[str, float]) -> ArticulationCfg:
    base_joint_pos = dict(UNITREE_G1_29DOF_CFG.init_state.joint_pos)
    for key in list(base_joint_pos.keys()):
        if any(token in key for token in ("shoulder", "elbow", "wrist")):
            del base_joint_pos[key]
    base_joint_pos.update(arm_hold_pose)
    return UNITREE_G1_29DOF_CFG.replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=UNITREE_G1_29DOF_CFG.init_state.pos,
            joint_pos=base_joint_pos,
            joint_vel=UNITREE_G1_29DOF_CFG.init_state.joint_vel,
        )
    )


def create_armhold_configs(arm_hold_pose: dict[str, float]):
    """Build train/play env cfg classes for a fixed arm hold pose."""
    robot_cfg = _build_robot_cfg(arm_hold_pose)

    @configclass
    class ArmHoldSceneCfg(base_cfg.RobotSceneCfg):
        robot: ArticulationCfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

    @configclass
    class ArmHoldActionsCfg(base_cfg.ActionsCfg):
        """Option A: policy controls lower body only; arms are held by fixed targets."""

        JointPositionAction = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=LOCOMOTION_JOINT_NAMES,
            scale=0.25,
            use_default_offset=True,
        )
        # Applied every physics substep (before sim); interval events alone run too late.
        ArmHoldPositionAction = mdp.ArmHoldPositionActionCfg(
            asset_name="robot",
            joint_names=ARM_JOINT_NAMES,
            target_positions=arm_hold_pose,
        )

    @configclass
    class ArmHoldEventCfg(base_cfg.EventCfg):
        hold_arms_reset = EventTerm(
            func=mdp.hold_joint_position_targets,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_positions": arm_hold_pose,
                "write_joint_state": True,
            },
        )
    @configclass
    class ArmHoldRewardsCfg(base_cfg.RewardsCfg):
        arm_pose_tracking = RewTerm(
            func=mdp.arm_pose_tracking_penalty,
            weight=-2.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_positions": arm_hold_pose,
            },
        )
        arm_joint_vel = RewTerm(
            func=mdp.arm_joint_vel_l2,
            weight=-0.01,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ARM_JOINT_NAMES,
                )
            },
        )
        arm_joint_torque = RewTerm(
            func=mdp.arm_joint_torque_l2,
            weight=-1.0e-4,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ARM_JOINT_NAMES,
                )
            },
        )

    @configclass
    class ArmHoldEnvCfg(base_cfg.RobotEnvCfg):
        scene: ArmHoldSceneCfg = ArmHoldSceneCfg(num_envs=4096, env_spacing=2.5)
        actions: ArmHoldActionsCfg = ArmHoldActionsCfg()
        events: ArmHoldEventCfg = ArmHoldEventCfg()
        rewards: ArmHoldRewardsCfg = ArmHoldRewardsCfg()

    @configclass
    class ArmHoldPlayEventCfg(ArmHoldEventCfg):
        hold_arms_interval = EventTerm(
            func=mdp.hold_joint_position_targets,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_positions": arm_hold_pose,
                "write_joint_state": True,
            },
        )

    @configclass
    class ArmHoldPlayEnvCfg(base_cfg.RobotPlayEnvCfg):
        scene: ArmHoldSceneCfg = ArmHoldSceneCfg(num_envs=32, env_spacing=2.5)
        actions: ArmHoldActionsCfg = ArmHoldActionsCfg()
        events: ArmHoldPlayEventCfg = ArmHoldPlayEventCfg()
        rewards: ArmHoldRewardsCfg = ArmHoldRewardsCfg()

        def __post_init__(self):
            super().__post_init__()
            self.commands.base_velocity.debug_vis = False

    return ArmHoldEnvCfg, ArmHoldPlayEnvCfg


ArmHoldEnvCfg, ArmHoldPlayEnvCfg = create_armhold_configs(ARM_HOLD_POSE)

ArmHoldGuardEnvCfg, ArmHoldGuardPlayEnvCfg = create_armhold_configs(POSE_PRESETS["guard"])
ArmHoldLowEnvCfg, ArmHoldLowPlayEnvCfg = create_armhold_configs(POSE_PRESETS["low"])
ArmHoldForwardEnvCfg, ArmHoldForwardPlayEnvCfg = create_armhold_configs(POSE_PRESETS["forward"])
ArmHoldTposeEnvCfg, ArmHoldTposePlayEnvCfg = create_armhold_configs(POSE_PRESETS["tpose"])
