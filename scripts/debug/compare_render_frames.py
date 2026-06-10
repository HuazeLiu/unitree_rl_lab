"""Save rgb_array frames at step 1 vs N; print mean pixel diff (detect frozen mesh in video)."""

import argparse
import pathlib
import sys

_script_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_script_root))
sys.path.insert(0, str(_script_root / "rsl_rl"))

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--compare_steps", type=int, default=100)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.checkpoint:
    parser.error("--checkpoint required")
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import gymnasium as gym

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


class _SyncRenderBeforeCaptureWrapper(gym.Wrapper):
    def _base_env(self):
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        return env

    def _sync_render(self) -> None:
        base = self._base_env()
        if not hasattr(base, "sim"):
            return
        sim = base.sim
        if sim.physics_sim_view is not None and sim.is_playing():
            sim.physics_sim_view.update_articulations_kinematic()
        sim.forward()
        sim.render()

    def render(self):
        self._sync_render()
        return self.env.render()

    def step(self, action):
        result = self.env.step(action)
        self._sync_render()
        return result


def main() -> None:
    use_fabric = not getattr(args_cli, "disable_fabric", False)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=use_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = _SyncRenderBeforeCaptureWrapper(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    while hasattr(base, "env"):
        base = base.env
    robot = base.scene["robot"]

    obs, _ = env.reset()
    p0 = robot.data.root_pos_w[0, :2].clone()
    frames: dict[int, np.ndarray] = {}
    n = args_cli.compare_steps
    for step in range(1, n + 1):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        if step in (1, n):
            frames[step] = env.render()
    disp = torch.norm(robot.data.root_pos_w[0, :2] - p0).item()
    diff = np.abs(frames[1].astype(np.float32) - frames[n].astype(np.float32)).mean()
    print(f"use_fabric={use_fabric} root_disp_xy={disp:.3f} mean_frame_diff={diff:.2f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
