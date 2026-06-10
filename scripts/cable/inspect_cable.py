#!/usr/bin/env python3
"""Inspect thick cable articulation DOFs in Isaac Sim."""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.sim import SimulationContext
from unitree_rl_lab.assets.cable.thick_cable_cfg import CABLE_LIKE_CFG
from cable.cable_preset_utils import log_cable_cfg
from cable.cable_validation_common import make_simulation_cfg, reset_cable_state, setup_base_scene

cfg = CABLE_LIKE_CFG.replace(prim_path="/World/Cable", init_pos=(-0.4, 0.0, 1.2))
sim = SimulationContext(make_simulation_cfg(cfg))
cable = setup_base_scene(sim, cfg)
sim.reset()
reset_cable_state(cable, cfg)
print("num_bodies:", cable.num_bodies)
print("num_joints:", cable.num_joints)
log_cable_cfg(cfg, argparse.Namespace(cable_preset="cable"), cable)
for _ in range(10):
    cable.write_data_to_sim()
    sim.step()
    cable.update(sim.get_physics_dt())
print("after 10 steps body_pos[0]:", cable.data.body_pos_w[0, 0].tolist())
simulation_app.close()
