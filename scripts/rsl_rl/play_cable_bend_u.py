#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Launch play.py for Unitree-G1-29dof-Cable-Bend-U with optional vision flags."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    play_script = Path(__file__).resolve().parent / "play.py"
    argv = list(sys.argv[1:])
    if "--task" not in argv:
        argv = ["--task", "Unitree-G1-29dof-Cable-Bend-U", *argv]
    if "--use_vision" in argv:
        argv = [a for a in argv if a != "--use_vision"]
        if "--enable_cameras" not in argv:
            argv.append("--enable_cameras")
    cmd = [sys.executable, str(play_script), *argv]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
