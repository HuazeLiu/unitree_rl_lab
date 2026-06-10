#!/usr/bin/env python3
"""Save high-resolution snapshots of Stage-2 Cable-Bend-U (cable welded to both hands).

The environment is stepped for a number of frames (legs locked, arms held at the cable-hold
pose, cable endpoints welded to the hands) so the fixed joints settle and the headless RTX
render reflects the true physical state. Endpoint-to-hand distances are printed as a
correctness check.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from isaaclab.app import AppLauncher

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("pip install pillow") from exc

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="demos/cable_bend_stage2_snapshot.png")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--width", type=int, default=1920, help="Render width (e.g. 1920 or 2560).")
parser.add_argument("--height", type=int, default=1080, help="Render height (e.g. 1080 or 1440).")
parser.add_argument("--steps", type=int, default=40, help="Physics steps before capture (settle).")
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
app = AppLauncher(args).app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import unitree_rl_lab.tasks  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

TASK = "Unitree-G1-29dof-Cable-Bend-U"

# (eye_offset, lookat_offset) relative to the robot root (root z ~0.8). The cable/hands sit
# ~0.15 m above the root, so we aim there and keep the eye near chest height for good framing.
VIEWS = {
    "iso": ((1.5, -1.3, 0.5), (0.30, 0.0, 0.18)),
    "front": ((2.1, 0.0, 0.35), (0.30, 0.0, 0.18)),
}


def _unwrap(env):
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    while hasattr(base, "env"):
        base = base.env
    return base


def _set_camera(base_env, eye_off, lookat_off) -> None:
    root = base_env.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
    base_env.sim.set_camera_view(eye=root + np.array(eye_off), target=root + np.array(lookat_off))


def _render(base_env) -> np.ndarray:
    """Render after pushing physics transforms to USD (headless fabric sync)."""
    import carb

    base_env.scene.write_data_to_sim()
    sim = base_env.sim
    sim.forward()
    if sim.is_fabric_enabled():
        sim._update_fabric(0.0, 0.0)
    carb_settings = carb.settings.get_settings()
    carb_settings.set_bool("/physics/updateToUsd", True)
    carb_settings.set_bool("/physics/fabricUpdateJointStates", True)
    sim.render()
    frame = base_env.render(recompute=False)
    carb_settings.set_bool("/physics/updateToUsd", False)
    if frame is None or frame.size == 0 or frame.max() == 0:
        sim.render()
        frame = base_env.render(recompute=False)
    if frame is None or frame.size == 0:
        raise RuntimeError("render returned an empty frame")
    return np.asarray(frame)


def _endpoint_to_hand_distances(base_env) -> tuple[float, float]:
    import importlib

    import isaaclab.utils.math as m

    bend_cfg = importlib.import_module(
        "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.g1_cable_bend_u_env_cfg"
    )
    ccfg = bend_cfg.G1_CABLE_BEND_SCENE_CABLE_CFG
    robot = base_env.scene["robot"]
    cable = base_env.scene["cable"]
    lid, _ = robot.find_bodies("left_rubber_hand")
    rid, _ = robot.find_bodies("right_rubber_hand")
    lp = robot.data.body_pos_w[0, lid[0]]
    rp = robot.data.body_pos_w[0, rid[0]]
    sid, _ = cable.find_bodies("seg_00")
    eid, _ = cable.find_bodies(f"seg_{ccfg.num_segments - 1:02d}")
    hs = 0.5 * ccfg.segment_length
    dev = base_env.device
    s_tip = cable.data.body_pos_w[0:1, sid[0]] + m.quat_apply(
        cable.data.body_quat_w[0:1, sid[0]], torch.tensor([[-hs, 0, 0]], device=dev)
    )
    e_tip = cable.data.body_pos_w[0:1, eid[0]] + m.quat_apply(
        cable.data.body_quat_w[0:1, eid[0]], torch.tensor([[hs, 0, 0]], device=dev)
    )
    start_err = float(torch.linalg.norm(s_tip[0] - lp))
    end_err = float(torch.linalg.norm(e_tip[0] - rp))
    return start_err, end_err


def main() -> None:
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    cfg = parse_env_cfg(TASK, num_envs=args.num_envs)
    cfg.viewer.resolution = (args.width, args.height)
    env = gym.make(TASK, cfg=cfg, render_mode="rgb_array")
    base = _unwrap(env)

    env.reset()
    action = torch.zeros(args.num_envs, base.action_manager.total_action_dim, device=base.device)
    for _ in range(args.steps):
        env.step(action)

    start_err, end_err = _endpoint_to_hand_distances(base)
    ok = max(start_err, end_err) < 0.02
    print(
        f"[CHECK] left-end->left-hand={start_err * 1000:.1f} mm, "
        f"right-end->right-hand={end_err * 1000:.1f} mm -> "
        f"{'OK (<20 mm)' if ok else 'WARNING (>=20 mm: weld may be wrong)'}",
        flush=True,
    )

    for name, (eye_off, lookat_off) in VIEWS.items():
        _set_camera(base, eye_off, lookat_off)
        frame = _render(base)
        view_path = out if name == "iso" else out.with_name(f"{out.stem}_{name}{out.suffix}")
        Image.fromarray(frame).save(view_path)
        print(f"[INFO] Saved {name} view {frame.shape[1]}x{frame.shape[0]} -> {view_path.resolve()}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    app.close()
