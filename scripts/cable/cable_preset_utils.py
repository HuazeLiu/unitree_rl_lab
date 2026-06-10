# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve cable presets and CLI overrides (import only after SimulationApp starts)."""

from __future__ import annotations

import argparse
import importlib

from unitree_rl_lab.assets.cable.thick_cable_cfg import ThickCableCfg

_CLI_OVERRIDE_FIELDS = (
    "num_segments",
    "total_mass",
    "bend_stiffness",
    "bend_damping",
    "twist_stiffness",
    "twist_damping",
    "bend_limit_deg",
    "max_joint_effort",
    "physics_dt",
)


def _preset_module():
    import unitree_rl_lab.assets.cable.thick_cable_cfg as mod

    return importlib.reload(mod)


def resolve_cable_preset(args: argparse.Namespace) -> ThickCableCfg:
    """Return base cfg for ``--cable_preset``, reloading thick_cable_cfg.py each run."""
    mod = _preset_module()
    presets = {
        "default": mod.THICK_CABLE_DEFAULT_CFG,
        "cable": mod.CABLE_LIKE_CFG,
        "rubber": mod.THICK_RUBBER_TUBE_CFG,
        "stiffer": mod.STIFFER_RUBBER_HOSE_CFG,
        "softer": mod.SOFTER_ROPELIKE_CABLE_CFG,
        "legacy": mod.LEGACY_FLEXIBLE_CABLE_CFG,
    }
    name = getattr(args, "cable_preset", "default")
    return presets.get(name, mod.THICK_CABLE_DEFAULT_CFG)


def apply_cable_cli_overrides(cfg: ThickCableCfg, args: argparse.Namespace) -> ThickCableCfg:
    updates: dict[str, object] = {}
    for field in _CLI_OVERRIDE_FIELDS:
        if hasattr(args, field):
            value = getattr(args, field)
            if value is not None:
                updates[field] = value
    return cfg.replace(**updates) if updates else cfg


def finalize_cable_cfg(args: argparse.Namespace, **scene_fields: object) -> ThickCableCfg:
    cfg = apply_cable_cli_overrides(resolve_cable_preset(args), args)
    return cfg.replace(**scene_fields) if scene_fields else cfg


def log_cable_cfg(cfg: ThickCableCfg, args: argparse.Namespace, cable: object | None = None) -> None:
    import unitree_rl_lab.assets.cable.thick_cable_cfg as mod

    preset = getattr(args, "cable_preset", "default")
    print(f"[INFO] cable cfg file: {mod.__file__}")
    print(
        f"[INFO] preset={preset!r}  segments={cfg.num_segments}  mass={cfg.total_mass:.3f} kg  "
        f"bend_k={cfg.bend_stiffness:.4f}  bend_d={cfg.bend_damping:.4f}  "
        f"bend_lim={cfg.bend_limit_deg:.1f} deg  effort_lim={cfg.max_joint_effort:.2f}  dt={cfg.physics_dt:.6f}"
    )
    if cable is not None and hasattr(cable, "data") and cable.data.joint_stiffness is not None:
        names = cable.joint_names
        stiffness = cable.data.joint_stiffness[0]
        twist = [float(stiffness[i]) for i, n in enumerate(names) if n.endswith(":0")][:3]
        bend = [float(stiffness[i]) for i, n in enumerate(names) if n.endswith((":1", ":2"))][:3]
        print(f"[INFO] sim stiffness twist DOFs (:0): {twist}  bend DOFs (:1,:2): {bend}")
