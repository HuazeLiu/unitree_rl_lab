from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def hold_joint_position_targets(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    target_positions: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    write_joint_state: bool = False,
) -> None:
    """Set fixed joint position targets for selected joints every event trigger.

    When ``write_joint_state`` is True (use on reset), also writes joint positions and zero
    velocities into the simulator so the held pose is visible immediately and is not pulled
    toward zero targets before the first interval event.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.int64)

    for joint_name, target_pos in target_positions.items():
        joint_ids, _ = asset.find_joints(joint_name)
        target = torch.full((len(env_ids), len(joint_ids)), target_pos, device=env.device)
        asset.set_joint_position_target(target, joint_ids=joint_ids, env_ids=env_ids)
        if write_joint_state:
            zero_vel = torch.zeros_like(target)
            asset.write_joint_state_to_sim(target, zero_vel, joint_ids=joint_ids, env_ids=env_ids)


def attach_g1_cable_to_hands(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Align the scene cable between G1 hands and create hand-to-cable fixed joints."""
    from unitree_rl_lab.assets.cable.g1_cable_attach import attach_cable_endpoints_to_g1
    from unitree_rl_lab.assets.cable.thick_cable_cfg import CABLE_LIKE_CFG

    attach_cable_endpoints_to_g1(env, env_ids, cable_hold_cfg=CABLE_LIKE_CFG)
