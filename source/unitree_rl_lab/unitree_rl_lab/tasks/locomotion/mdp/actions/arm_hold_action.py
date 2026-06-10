"""Fixed arm pose action term (zero policy dims, applied every physics substep)."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm
from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def _ensure_dof_buffers_writable(asset: Articulation) -> None:
    """Clone DOF buffers after inference_mode steps so Isaac Lab can inplace-write joint state."""
    data = asset._data
    for buf_attr in ("_joint_pos", "_joint_vel", "_joint_acc", "_previous_joint_vel"):
        buf = getattr(data, buf_attr, None)
        if buf is None or buf.data is None:
            continue
        if buf.data.is_inference():
            buf.data = buf.data.clone()


class ArmHoldPositionAction(ActionTerm):
    """Apply fixed joint position targets before every physics step.

    Interval events run after physics and cannot prevent early substeps from
    driving uncontrolled joints toward zero targets. This term keeps arm targets
    locked while the policy only commands the legs (action_dim stays unchanged).
    """

    cfg: ArmHoldPositionActionCfg

    def __init__(self, cfg: ArmHoldPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        targets = [cfg.target_positions[name] for name in self._joint_names]
        self._targets = torch.tensor(targets, device=env.device, dtype=torch.float32).unsqueeze(0)
        self._targets = self._targets.repeat(env.num_envs, 1)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return torch.empty(self.num_envs, 0, device=self.device)

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        del actions

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._targets, joint_ids=self._joint_ids)
        if self.cfg.kinematic_lock:
            _ensure_dof_buffers_writable(self._asset)
            pos = self._targets.detach().clone()
            zero_vel = torch.zeros_like(pos)
            self._asset.write_joint_state_to_sim(pos, zero_vel, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        del env_ids


@configclass
class ArmHoldPositionActionCfg(ActionTermCfg):
    """Configuration for a fixed arm-hold action term (no policy dimensions)."""

    class_type: type[ActionTerm] = ArmHoldPositionAction

    joint_names: list[str] = MISSING
    """Joint names to hold, in the order used for target lookup."""

    target_positions: dict[str, float] = MISSING
    """Desired position target for each held joint."""

    kinematic_lock: bool = True
    """If True, write arm joint state each substep (required for guard pose at play/train)."""
