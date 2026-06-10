# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_width", type=int, default=None, help="Viewport width for video recording.")
parser.add_argument("--video_height", type=int, default=None, help="Viewport height for video recording.")
parser.add_argument("--video_dir", type=str, default=None, help="Output directory for recorded videos.")
parser.add_argument(
    "--video_name_prefix",
    type=str,
    default="rl-video",
    help="Filename prefix for recorded videos.",
)
parser.add_argument(
    "--demo_multi_env",
    action="store_true",
    default=False,
    help="Use a wide world camera framing all parallel envs (for demo videos).",
)
parser.add_argument(
    "--skip_export",
    action="store_true",
    default=False,
    help="Skip JIT/ONNX policy export (faster for video-only runs).",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _get_manager_env(env):
    """Unwrap gym/RslRl/RecordVideo wrappers to ManagerBasedRLEnv."""
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    while hasattr(base, "env"):
        base = base.env
    return base


def _sync_viewport_camera(base_env) -> None:
    """Point /OmniverseKit_Persp at the robot (same logic as GUI play / ViewportCameraController)."""
    import numpy as np

    vcc = getattr(base_env, "viewport_camera_controller", None)
    if vcc is not None:
        if base_env.cfg.viewer.origin_type == "world":
            vcc.update_view_to_world()
        elif base_env.cfg.viewer.origin_type == "env":
            vcc.update_view_to_env()
        elif base_env.cfg.viewer.asset_name:
            if base_env.cfg.viewer.origin_type == "asset_body" and base_env.cfg.viewer.body_name:
                vcc.update_view_to_asset_body(base_env.cfg.viewer.asset_name, base_env.cfg.viewer.body_name)
            else:
                vcc.update_view_to_asset_root(base_env.cfg.viewer.asset_name)
        vcc.update_view_location()
        return
    robot = base_env.scene["robot"]
    origin = robot.data.root_pos_w[0].detach().cpu().numpy()
    eye = origin + np.array(base_env.cfg.viewer.eye, dtype=float)
    target = origin + np.array(base_env.cfg.viewer.lookat, dtype=float)
    base_env.sim.set_camera_view(eye=eye, target=target)


def _apply_armhold_for_render(base_env) -> None:
    """Sync guard arm pose into PhysX + fabric before RTX capture (legs keep sim state)."""
    from unitree_rl_lab.tasks.locomotion.mdp.actions.arm_hold_action import _ensure_dof_buffers_writable

    try:
        arm_term = base_env.action_manager.get_term("ArmHoldPositionAction")
    except KeyError:
        return
    robot = base_env.scene["robot"]
    arm_term.apply_actions()
    # Partial arm writes update PhysX DOFs but Hydra/fabric skinning can stay at URDF hang.
    # Patch arms in a full joint snapshot (legs from sim, arms from hold targets) then push all DOFs.
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = robot.data.joint_vel.clone()
    joint_ids = arm_term._joint_ids
    joint_pos[:, joint_ids] = arm_term._targets
    joint_vel[:, joint_ids] = 0.0
    _ensure_dof_buffers_writable(robot)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(arm_term._targets, joint_ids=joint_ids)


def _capture_rgb_frame(base_env) -> "np.ndarray":
    """Capture rgb from viewport camera after arm-hold + fabric/USD sync."""
    import numpy as np

    _apply_armhold_for_render(base_env)
    _sync_viewport_camera(base_env)
    base_env.scene.write_data_to_sim()
    sim = base_env.sim
    sim.forward()
    if sim.is_fabric_enabled():
        sim._update_fabric(0.0, 0.0)

    # Draw cable bend visualization (target U, current centerline) before render.
    _update_cable_bend_viz(base_env)

    # Push joint transforms to Hydra for URDF visuals (headless kit keeps updateToUsd=false by default).
    import carb

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
        res = base_env.cfg.viewer.resolution
        return np.zeros((res[1], res[0], 3), dtype=np.uint8)
    return frame


def _update_cable_bend_viz(base_env) -> None:
    """Draw target U-shape, current cable centerline, and endpoints in the viewport.

    Only activates for Cable-Bend-U tasks (checks for cable + cable_bend_u command).
    """
    try:
        base_env.scene["cable"]
    except KeyError:
        return
    try:
        from unitree_rl_lab.assets.cable.thick_cable_viz import ThickCableVisualizer
        from unitree_rl_lab.tasks.locomotion.mdp.cable.commands import get_cable_bend_command
    except ImportError:
        return

    cable = base_env.scene["cable"]
    try:
        cmd = get_cable_bend_command(base_env, "cable_bend_u")
    except Exception:
        return

    if not hasattr(base_env, "_cable_viz"):
        base_env._cable_viz = ThickCableVisualizer(
            centerline_path="/Visuals/CableCenterline",
            target_path="/Visuals/CableTarget",
            endpoint_path="/Visuals/CableEndpoints",
        )

    viz = base_env._cable_viz
    viz.draw_target_curve(cmd.target_centerline, env_id=0)
    viz.draw_centerline(cable, env_id=0)
    viz.draw_endpoint_frames(cable, env_id=0)


def _save_mp4(frames: list, out_path: str, fps: float) -> None:
    import imageio.v2 as imageio

    imageio.mimsave(out_path, frames, fps=fps)
    print(f"[INFO] Wrote video ({len(frames)} frames, {fps:.1f} fps): {out_path}")


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    use_fabric = not args_cli.disable_fabric
    if args_cli.video and not args_cli.disable_fabric:
        print("[INFO] play/video: viewport capture + fabric/USD arm sync (use_fabric=True).")
    elif args_cli.video:
        print("[INFO] play/video: viewport capture with use_fabric=False (USD arm visuals, slower).")
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=use_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.demo_multi_env and num_envs > 1:
        spacing = env_cfg.scene.env_spacing
        cols = math.ceil(math.sqrt(num_envs))
        rows = math.ceil(num_envs / cols)
        center_x = spacing * (cols - 1) * 0.5
        center_y = spacing * (rows - 1) * 0.5
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.eye = (center_x + 7.0, center_y + 7.0, 4.5)
        env_cfg.viewer.lookat = (center_x, center_y, 0.85)

    if args_cli.video:
        width = args_cli.video_width or 1920
        height = args_cli.video_height or 1080
        env_cfg.viewer.resolution = (width, height)
        if not args_cli.demo_multi_env and num_envs == 1:
            env_cfg.viewer.origin_type = "asset_root"
            env_cfg.viewer.asset_name = "robot"
            env_cfg.viewer.env_index = 0
            env_cfg.viewer.eye = (2.5, 2.5, 1.8)
            env_cfg.viewer.lookat = (0.0, 0.0, 0.85)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    video_folder = None
    if args_cli.video:
        video_folder = args_cli.video_dir or os.path.join(log_dir, "videos", "play")
        video_folder = os.path.abspath(video_folder)
        os.makedirs(video_folder, exist_ok=True)
        print(f"[INFO] Recording video (manual capture) -> {video_folder}")

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    if not args_cli.skip_export:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # Reset is required: spawn leaves arms at URDF zero until reset events apply the hold pose.
    obs, _ = env.reset()
    base_env = _get_manager_env(env)
    video_frames: list = []
    if args_cli.video:
        robot = base_env.scene["robot"]
        jid, _ = robot.find_joints("left_elbow_joint")
        print(
            f"[INFO] After reset: left_elbow={robot.data.joint_pos[0, jid[0]].item():.3f} "
            f"(guard target 1.15)"
        )
        for _ in range(3):
            base_env.sim.render()
        video_frames.append(_capture_rgb_frame(base_env))

    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # Video capture calls write_joint_state_to_sim after step; inference_mode marks DOF
        # buffers as inference tensors and forbids inplace updates outside that context.
        step_ctx = torch.no_grad() if args_cli.video else torch.inference_mode()
        with step_ctx:
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        if base_env.sim.has_gui():
            if base_env.sim.physics_sim_view is not None and base_env.sim.is_playing():
                base_env.sim.physics_sim_view.update_articulations_kinematic()
            base_env.sim.forward()
            base_env.sim.render()
        if args_cli.video:
            video_frames.append(_capture_rgb_frame(base_env))
            timestep += 1
            if timestep >= args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if args_cli.video and video_frames:
        video_path = os.path.join(video_folder, f"{args_cli.video_name_prefix}-step-0.mp4")
        _save_mp4(video_frames, video_path, fps=max(1.0 / dt, 1.0))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
