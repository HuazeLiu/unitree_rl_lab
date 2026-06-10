# Thick Cable Simulator (Articulated Rigid-Body Approximation)

This module implements a **thick cable** for humanoid manipulation / RL using a chain of short **capsule rigid bodies** connected by passive **D6 joints**. It is **not** a material-accurate cable model and **does not** use FEM deformables.

## G1 cable-hold scene

G1 stands on flat ground with arms raised; cable endpoints are welded to `left_rubber_hand` / `right_rubber_hand`.

```bash
conda activate unitree_g1_lab
cd unitree_rl_lab
# GUI: runs until you close the Isaac Sim window
python scripts/cable/play_g1_cable_hold.py --slowdown 0.01
# Headless smoke test (must set --num_steps)
python scripts/cable/play_g1_cable_hold.py --headless --num_steps 200
```

Gym task: `Unitree-G1-29dof-Cable-Hold` (config: `g1_cable_hold_env_cfg.py`).

## G1 cable U-bend RL (whole-body manipulation)

Train a G1-29dof policy to bend a held cable into one of **four** commanded U depths. Training uses **privileged cable geometry** in the policy observation (no RGB camera). Deploy / play can use a **ZED-style head camera** plus **SAM** segmentation (`unitree_rl_lab.perception`).

| Gym ID | Stage | Description |
|--------|-------|-------------|
| `Unitree-G1-29dof-Cable-Bend-U` | **bend** (default) | Cable welded to both hands at reset; learn U-bend |
| `Unitree-G1-29dof-Cable-Bend-U-Full` | **full** | Cable on table; reach, grasp, attach, then bend |

Config: `tasks/locomotion/robots/g1/29dof/g1_cable_bend_u_env_cfg.py`. PPO: `agents/rsl_rl_cable_bend_ppo_cfg.py`.

### Policy observation layout (no images)

| Term | Dim |
|------|-----|
| `base_ang_vel` | 3 |
| `projected_gravity` | 3 |
| `joint_pos_rel` / `joint_vel_rel` | 29 + 29 |
| `last_action` | action_dim |
| `cable_bend_command` (one-hot depth) | 4 |
| `cable_endpoints_body` | 6 |
| `cable_centerline_sparse` | 24 |
| `grasp_progress` | 2 |
| `phase_progress` | 1 |

**Stacked dims (history_length=5):** policy **650**, critic **1365**.  
**Action dim:** 29 (whole-body joint position targets, scale 0.1).

**Critic-only extra terms (single step):** `base_lin_vel` (3), `cable_centerline_full_body` (72 = 24 segments × 3), `hand_pos_body` (6), `hand_contact_force` (2), `target_centerline_flat` (60 = 20 target points × 3).

### Training stability notes

If PPO crashes with `normal expects all elements of std >= 0.0`:

1. Use updated code (`noise_std_type=log`, `CableBendRslRlVecEnvWrapper`, action scale 0.1).
2. Start with `--num_envs 4096` if 8192 is unstable; scale up after smoke.
3. Smoke: `python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Cable-Bend-U --num_envs 64 --max_iterations 5`

### Training (8192 envs, headless, no camera)

```bash
conda activate unitree_g1_lab
cd unitree_rl_lab
pip install -e source/unitree_rl_lab/

# Smoke
python scripts/rsl_rl/train.py --headless \
  --task Unitree-G1-29dof-Cable-Bend-U \
  --num_envs 64 --max_iterations 100 --run_name smoke_bend

# Production
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Cable-Bend-U \
  --num_envs 8192 --max_iterations 20000 \
  --run_name cable_bend_u_8192x20k --seed 42
```

Stage-1 (pick from table):

```bash
python scripts/rsl_rl/train.py --headless \
  --task Unitree-G1-29dof-Cable-Bend-U-Full \
  --num_envs 8192 --max_iterations 30000 \
  --run_name cable_bend_u_full_8192
```

### Play / vision

```bash
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Cable-Bend-U \
  --num_envs 4 --checkpoint logs/rsl_rl/unitree_g1_29dof_cable_bend_u/.../model_xxxx.pt

# Optional ZED + SAM (play env enables head camera)
python scripts/rsl_rl/play_cable_bend_u.py --use_vision --enable_cameras \
  --checkpoint logs/rsl_rl/.../model_xxxx.pt
```

SAM offline demo:

```bash
pip install -e "source/unitree_rl_lab[perception]"
python scripts/perception/demo_sam_cable.py --image frame.png --checkpoint sam_vit_b.pth
```

U-target unit test (no sim):

```bash
python source/unitree_rl_lab/unitree_rl_lab/assets/cable/test_u_shape_targets.py
```

## Quick start

Requires the `unitree_g1_lab` conda environment (Isaac Sim 5.1):

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unitree_g1_lab
cd unitree_rl_lab
pip install -e source/unitree_rl_lab/

python scripts/cable/validate_sag_test.py --headless
python scripts/cable/validate_push_on_table.py --headless
python scripts/cable/validate_torso_support.py --headless

# Batch validation logs
bash scripts/cable/record_cable_validations.sh
```

## Visual inspection and video recording

### Interactive GUI

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unitree_g1_lab
cd unitree_rl_lab
python scripts/cable/visualize_thick_cable.py --scene sag --num_steps 1000 --slowdown 0.01
python scripts/cable/visualize_thick_cable.py --scene push --num_steps 1000 --slowdown 0.01
python scripts/cable/visualize_thick_cable.py --scene torso --num_steps 1000 --slowdown 0.01
```

Omit `--headless` to open the Isaac Sim viewer. Use `--slowdown` to pause between physics steps for inspection.

Individual validation scripts also support the same visual flags:

```bash
python scripts/cable/validate_sag_test.py --num_steps 600 --slowdown 0.01
python scripts/cable/validate_push_on_table.py --enable_cameras
python scripts/cable/validate_torso_support.py --record_video --headless --enable_cameras
```

Shared CLI options: `--headless`, `--enable_cameras`, `--record_video`, `--video_dir demos/thick_cable/videos`, `--num_steps`, `--slowdown`.

### Headless recording

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unitree_g1_lab
cd unitree_rl_lab
python scripts/cable/visualize_thick_cable.py --scene sag --headless --enable_cameras --record_video
python scripts/cable/visualize_thick_cable.py --scene push --headless --enable_cameras --record_video
python scripts/cable/visualize_thick_cable.py --scene torso --headless --enable_cameras --record_video
```

PNG frames are written under `demos/thick_cable/videos/{sag,push,torso}/frame_XXXXXX.png`. Convert to mp4:

```bash
python scripts/cable/frames_to_video.py --video_dir demos/thick_cable/videos
```

Output: `demos/thick_cable/videos/sag.mp4`, `push.mp4`, `torso.mp4`.

`--scene all` in headless mode launches each scene in a separate subprocess (Isaac Sim cannot reliably rebuild multiple scenes in one process).

## API

### Configuration

- `ThickCableCfg` — full cable + simulation parameters
- `CABLE_LIKE_CFG` / `THICK_CABLE_DEFAULT_CFG` — inextensible elastic-rod preset (24 segments, separate bend/twist drives)
- `THICK_RUBBER_TUBE_CFG` — stiffer 14-segment hose (previous default tuning)
- `LEGACY_FLEXIBLE_CABLE_CFG` — previous 24-segment compliant defaults
- `SOFTER_ROPELIKE_CABLE_CFG` — more compliant than default
- `STIFFER_RUBBER_HOSE_CFG` — industrial hose (12 segments, very stiff)

### Builder

```python
from unitree_rl_lab.assets.cable import create_thick_cable, THICK_CABLE_DEFAULT_CFG, make_thick_cable_articulation_cfg

create_thick_cable(
    prim_path="/World/Cable",
    total_length=0.80,
    radius=0.025,
    num_segments=24,
    total_mass=0.8,
)

articulation_cfg = make_thick_cable_articulation_cfg(THICK_CABLE_DEFAULT_CFG)
```

### Utilities (RL rewards / observations)

| Function | Purpose |
|----------|---------|
| `get_cable_segment_poses()` | Per-segment position + orientation |
| `get_cable_centerline_points()` | Centerline polyline |
| `get_cable_endpoint_poses()` | First/last segment poses |
| `compute_centerline_chamfer_reward()` | Soft chamfer vs target curve |
| `compute_endpoint_reward()` | Endpoint pose matching |
| `compute_bending_energy()` | Sum of squared joint positions |
| `compute_smoothness_reward()` | Penalize jagged centerline |

### Debug visualization

`ThickCableVisualizer` draws centerline, target curve, endpoint frames, and optional curvature coloring.

## Default physical parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `total_length` | 0.80 m | |
| `radius` | 0.025 m | Visual = collision |
| `num_segments` | 16 | Fewer links → less local snake / waviness |
| `total_mass` | 1.2 kg | Higher inertia, resists side push |
| `bend_limit_deg` | ±7° | Larger minimum bend radius per link |
| `twist_limit_deg` | ±3° | Per joint |
| `bend_stiffness` | 0.28 N·m/rad | Rubber-tube bending resistance |
| `bend_damping` | 0.045 N·m·s/rad | Smoother centerline |
| `twist_stiffness` | 0.10 N·m/rad | Reduces corkscrew snake |
| `twist_damping` | 0.025 N·m·s/rad | |
| `max_joint_effort` | 2.5 N·m | Passive drive force limit |
| `static_friction` | 0.9 | |
| `dynamic_friction` | 0.8 | |
| `restitution` | 0.0 | |
| `contact_offset` | 0.003 m | |
| `self_collision` | False | Enable only for qualitative tests |
| `physics_dt` | 1/120 s | |
| `position_solver_iterations` | 20 | Articulation root |
| `velocity_solver_iterations` | 6 | |

## Joint model

- **D6 joints** between adjacent capsules
- **Translation locked** (`transX/Y/Z` limits with `low > high`)
- **Bending** on `rotY` and `rotZ` with symmetric stiffness/damping (PhysX recommendation)
- **Twist** on `rotX` with smaller limit and lower stiffness
- **No actuated targets** — drives use zero target position/velocity for passive resistance only
- Isaac Lab `ImplicitActuatorCfg` applies **bend** to DOFs ``joint_XX:1`` and ``joint_XX:2``, **twist** to ``joint_XX:0`` (PhysX D6 naming)

## Tuning protocol

Run **`validate_sag_test.py` first**. Tune in this order:

1. `num_segments`
2. `segment_mass` / `total_mass`
3. `bend_limit_deg`
4. `bend_stiffness`
5. `bend_damping`
6. Contact friction
7. Solver iterations
8. `self_collision=True` only after basic stability

### Symptom → action

| Symptom | Check |
|---------|-------|
| Behaves like one rigid rod | Rotational axes not limited/unlocked correctly |
| Springy rubber band | Translations not locked; damping too low |
| Explodes / jitter | ↓ stiffness, ↑ damping, ↓ dt, ↓ `num_segments` |
| Collapses like string | ↑ stiffness or ↓ bend limit |
| Won't bend under contact | ↓ stiffness or ↑ bend limit |

### Parameter sweep

```bash
../IsaacLab/isaaclab.sh -p scripts/cable/run_parameter_sweep.py --headless --max_runs 8
```

Grid:

- `bend_stiffness`: [0.03, 0.08, 0.15, 0.30]
- `bend_damping`: [0.006, 0.012, 0.025, 0.05]
- `bend_limit_deg`: [8, 12, 18, 25]
- `num_segments`: [16, 24, 32]
- `total_mass`: [0.4, 0.8, 1.2]

Select defaults when:

- No visible stretch
- No violent jitter
- Bends under hand/torso contact
- Does not collapse like a string or behave like a rod
- Settles without long oscillation
- Centerline smooth enough for RL rewards

## Validation scenes

| Script | Scenario | Output |
|--------|----------|--------|
| `validate_sag_test.py` | Fixed endpoints, gravity sag | `demos/thick_cable/sag_test/sag_centerline.jsonl` |
| `validate_push_on_table.py` | Table + kinematic pusher | `demos/thick_cable/push_on_table/` |
| `validate_torso_support.py` | Torso-like cylinder contact | `demos/thick_cable/torso_support/` |

## Presets

```python
from unitree_rl_lab.assets.cable import (
    CABLE_LIKE_CFG,
    THICK_CABLE_DEFAULT_CFG,
    THICK_RUBBER_TUBE_CFG,
    LEGACY_FLEXIBLE_CABLE_CFG,
    SOFTER_ROPELIKE_CABLE_CFG,
    STIFFER_RUBBER_HOSE_CFG,
)
```

**Default (`CABLE_LIKE_CFG`)**: 24 segments, 0.95 kg, ±4.5° bend / ±1.2° twist per link, 240 Hz physics.

CLI: `--cable_preset cable` (same as default) | `rubber` | `stiffer` | `softer` | `legacy`.

**LEGACY_FLEXIBLE_CABLE_CFG**: old 24-segment / 0.8 kg settings (more local waviness).

**SOFTER_ROPELIKE_CABLE_CFG**: 20 segments, moderate compliance.

**STIFFER_RUBBER_HOSE_CFG**: 12 segments, 1.4 kg, ±5° bend, highest stiffness.

## File layout

```
unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/cable/
  thick_cable_cfg.py       # ThickCableCfg + presets
  thick_cable_builder.py   # create_thick_cable (USD D6 chain)
  thick_cable_utils.py     # poses + rewards
  thick_cable_viz.py       # debug drawing

unitree_rl_lab/scripts/cable/
  validate_sag_test.py
  validate_push_on_table.py
  validate_torso_support.py
  visualize_thick_cable.py
  frames_to_video.py
  cable_visual_runtime.py
  cable_scenes.py
  cable_validation_common.py
  run_parameter_sweep.py
  record_cable_validations.sh
```

## Limitations

- Articulated rigid-body approximation only; not material-accurate.
- D6 drive behavior may vary slightly across Isaac Sim versions — always confirm bending visually with `validate_sag_test.py`.
- Isaac Lab `Articulation` is primarily tested with revolute/prismatic joints; this cable uses composed D6 limits/drives via raw USD.
