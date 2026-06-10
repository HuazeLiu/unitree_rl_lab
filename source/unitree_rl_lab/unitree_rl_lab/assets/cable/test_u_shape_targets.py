# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight tests for U-shape target generation (no Isaac Sim)."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import torch

_u_path = pathlib.Path(__file__).resolve().parent / "u_shape_targets.py"
_spec = importlib.util.spec_from_file_location("u_shape_targets_test", _u_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["u_shape_targets_test"] = _mod
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

from u_shape_targets_test import (  # type: ignore[import-not-found]
    NUM_U_DEPTH_PRESETS,
    U_SHAPE_DEPTH_PRESETS,
    depth_id_one_hot,
    generate_u_shape_centerline_batch,
    generate_u_shape_centerline_local,
)


def test_local_curve_shape() -> None:
    curve = generate_u_shape_centerline_local(U_SHAPE_DEPTH_PRESETS[0], num_points=20)
    assert curve.shape == (20, 3)
    assert curve[0, 0] < 0.0 < curve[-1, 0]
    assert curve[10, 2] < curve[0, 2]


def test_batch_world_frames() -> None:
    n = 4
    depth_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    torso_pos = torch.zeros(n, 3)
    torso_pos[:, 0] = 1.0
    torso_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(n, 1)
    world = generate_u_shape_centerline_batch(depth_ids, torso_pos, torso_quat, num_points=16)
    assert world.shape == (n, 16, 3)
    assert world[0, 10, 2] < world[0, 0, 2]


def test_one_hot() -> None:
    oh = depth_id_one_hot(torch.tensor([0, 3]))
    assert oh.shape == (2, NUM_U_DEPTH_PRESETS)
    assert oh[0, 0] == 1.0
    assert oh[1, 3] == 1.0


if __name__ == "__main__":
    test_local_curve_shape()
    test_batch_world_frames()
    test_one_hot()
    print("u_shape_targets: OK")
