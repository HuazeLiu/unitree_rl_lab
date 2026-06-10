#!/usr/bin/env python3
"""Extract key training metrics at selected iterations from rsl_rl train logs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


BLOCK_RE = re.compile(
    r"Learning iteration (\d+)/(\d+).*?"
    r"Mean reward:\s*([-\d.]+).*?"
    r"Mean episode length:\s*([-\d.]+).*?"
    r"Episode_Reward/track_lin_vel_xy:\s*([-\d.]+).*?"
    r"(?:Episode_Reward/arm_pose_tracking:\s*([-\d.]+).*?)?"
    r"Episode_Termination/bad_orientation:\s*([-\d.]+).*?"
    r"(?:Computation:\s*(\d+) steps/s.*?)?"
    r"(?:Total timesteps:\s*(\d+).*?)?",
    re.DOTALL,
)


def parse_log(path: Path, iters: list[int]) -> dict[int, dict[str, str | None]]:
    text = path.read_text(errors="replace")
    blocks: dict[int, dict[str, str | None]] = {}
    for match in BLOCK_RE.finditer(text):
        it = int(match.group(1))
        if it not in iters:
            continue
        blocks[it] = {
            "max_iter": match.group(2),
            "mean_reward": match.group(3),
            "episode_length": match.group(4),
            "track_lin_vel_xy": match.group(5),
            "arm_pose_tracking": match.group(6),
            "bad_orientation": match.group(7),
            "fps": match.group(8),
            "timesteps": match.group(9),
        }
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=Path)
    parser.add_argument("--iters", type=int, nargs="+", default=[0, 100, 199])
    args = parser.parse_args()

    if not args.log_file.exists():
        print(f"ERROR: log not found: {args.log_file}", file=sys.stderr)
        sys.exit(1)

    blocks = parse_log(args.log_file, args.iters)
    for it in args.iters:
        if it not in blocks:
            print(f"iter {it}: NOT FOUND")
            continue
        b = blocks[it]
        arm = b["arm_pose_tracking"]
        arm_str = f" arm_pose={arm}" if arm is not None else ""
        fps_str = f" fps={b['fps']}" if b["fps"] else ""
        print(
            f"iter {it}: reward={b['mean_reward']} ep_len={b['episode_length']} "
            f"track_xy={b['track_lin_vel_xy']}{arm_str} bad_ori={b['bad_orientation']}{fps_str}"
        )


if __name__ == "__main__":
    main()
