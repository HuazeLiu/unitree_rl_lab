# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Camera, recording, and simulation-loop helpers for thick cable visual workflows."""

from __future__ import annotations

import pathlib
import time
from typing import Callable

import isaacsim.core.utils.prims as prim_utils
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationContext

from cable.cable_visual_cli import SceneCameraView


class CableFrameRecorder:
    """Optional Isaac Lab RGB camera for frame capture."""

    def __init__(
        self,
        sim: SimulationContext,
        view: SceneCameraView,
        enabled: bool,
        record: bool,
        frame_dir: pathlib.Path,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self.sim = sim
        self.view = view
        self.record = record
        self.frame_dir = frame_dir
        self.camera: Camera | None = None
        self._frame_idx = 0

        if not enabled:
            return

        if record:
            if frame_dir.exists():
                for old in frame_dir.glob("frame_*.png"):
                    old.unlink()
            frame_dir.mkdir(parents=True, exist_ok=True)

        if not prim_utils.is_prim_path_valid("/World/CameraRig"):
            prim_utils.create_prim("/World/CameraRig", "Xform")

        camera_cfg = CameraCfg(
            prim_path="/World/CameraRig/SceneCamera",
            update_period=0,
            height=height,
            width=width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
        )
        self.camera = Camera(cfg=camera_cfg)

    def setup_view(self) -> None:
        if self.camera is None:
            return
        self.camera.reset()
        eye = torch.tensor([self.view.eye], device=self.sim.device, dtype=torch.float32)
        target = torch.tensor([self.view.target], device=self.sim.device, dtype=torch.float32)
        self.camera.set_world_poses_from_view(eye, target)

    def capture(self, step: int, stride: int) -> None:
        if self.camera is None or not self.record or step % stride != 0:
            return
        self.sim.render()
        self.camera.update(self.sim.get_physics_dt())
        if "rgb" not in self.camera.data.output:
            return
        rgb = self.camera.data.output["rgb"][0].detach().cpu().numpy()
        if rgb.ndim == 3 and rgb.shape[-1] > 3:
            rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        out_path = self.frame_dir / f"frame_{self._frame_idx:06d}.png"
        try:
            import imageio.v3 as iio

            iio.imwrite(out_path, rgb)
        except ImportError:
            from PIL import Image

            Image.fromarray(rgb).save(out_path)
        self._frame_idx += 1

    @property
    def frame_count(self) -> int:
        return self._frame_idx


def configure_viewport(sim: SimulationContext, view: SceneCameraView, headless: bool) -> None:
    """Set the Isaac Sim viewport camera when a GUI is available."""
    if not headless:
        sim.set_camera_view(list(view.eye), list(view.target))


def run_simulation_loop(
    sim: SimulationContext,
    simulation_app,
    num_steps: int,
    step_fn: Callable[[int], None],
    *,
    headless: bool,
    slowdown: float = 0.0,
    recorder: CableFrameRecorder | None = None,
    video_stride: int = 2,
    on_log_step: Callable[[int], None] | None = None,
    log_interval: int = 30,
) -> None:
    """Run the simulation for a fixed number of steps, optionally recording frames."""
    if recorder is not None:
        recorder.setup_view()

    for step in range(num_steps):
        step_fn(step)

        if recorder is not None:
            recorder.capture(step, video_stride)

        if on_log_step is not None and step % log_interval == 0:
            on_log_step(step)

        if not headless and not simulation_app.is_running():
            break

        if slowdown > 0.0 and not headless:
            time.sleep(slowdown)

    if headless and recorder is not None and recorder.record:
        print(f"[RESULT] saved {recorder.frame_count} frames to {recorder.frame_dir}")
