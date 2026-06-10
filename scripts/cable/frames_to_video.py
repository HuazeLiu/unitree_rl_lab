#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Convert saved PNG frame folders into mp4 videos."""

from __future__ import annotations

import argparse
import pathlib
import sys


def _load_writer():
    try:
        import imageio.v2 as imageio

        return "imageio", imageio
    except ImportError:
        pass
    try:
        import cv2

        return "opencv", cv2
    except ImportError:
        raise SystemExit("Install imageio or opencv-python to encode videos.")


def frames_to_mp4(frames_dir: pathlib.Path, output_path: pathlib.Path, fps: float) -> int:
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        print(f"[WARN] no frames in {frames_dir}")
        return 0

    backend, lib = _load_writer()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if backend == "imageio":
        writer = lib.get_writer(str(output_path), fps=fps, codec="libx264", quality=8)
        for frame_path in frames:
            writer.append_data(lib.imread(frame_path))
        writer.close()
    else:
        first = lib.imread(str(frames[0]))
        height, width = first.shape[:2]
        fourcc = lib.VideoWriter_fourcc(*"mp4v")
        writer = lib.VideoWriter(str(output_path), fourcc, fps, (width, height))
        writer.write(first)
        for frame_path in frames[1:]:
            writer.write(lib.imread(str(frame_path)))
        writer.release()

    return len(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert thick cable PNG frames to mp4.")
    parser.add_argument(
        "--video_dir",
        type=str,
        default="demos/thick_cable/videos",
        help="Root directory containing sag/, push/, torso/ frame folders.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Output video frame rate.")
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["sag", "push", "torso"],
        help="Scene subfolders to convert.",
    )
    args = parser.parse_args()

    root = pathlib.Path(args.video_dir)
    if not root.is_absolute():
        root = pathlib.Path(__file__).resolve().parents[2] / root

    for scene in args.scenes:
        frames_dir = root / scene
        out_path = root / f"{scene}.mp4"
        count = frames_to_mp4(frames_dir, out_path, args.fps)
        if count:
            print(f"[RESULT] {count} frames -> {out_path}")


if __name__ == "__main__":
    main()
