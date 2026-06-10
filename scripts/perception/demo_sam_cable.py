#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Offline SAM cable segmentation demo (no Isaac Sim required)."""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("pip install pillow") from exc

from unitree_rl_lab.perception import SamCableSegmentor, SamCableSegmentorCfg, mask_to_centerline_2d


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment cable in a single image with SAM.")
    parser.add_argument("--image", type=str, required=True, help="Input RGB image path.")
    parser.add_argument("--checkpoint", type=str, required=True, help="SAM checkpoint path.")
    parser.add_argument("--model-type", type=str, default="vit_b", help="SAM backbone type.")
    parser.add_argument("--output", type=str, default="", help="Optional mask output PNG.")
    args = parser.parse_args()

    rgb = np.array(Image.open(args.image).convert("RGB"))
    segmentor = SamCableSegmentor(
        SamCableSegmentorCfg(model_type=args.model_type, checkpoint_path=args.checkpoint)
    )
    mask = segmentor.segment(rgb)
    centerline = mask_to_centerline_2d(mask, num_points=8)
    print(f"[INFO] mask coverage: {mask.mean():.4f}")
    print(f"[INFO] centerline pixels:\n{centerline}")

    if args.output:
        out = pathlib.Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((mask * 255).astype(np.uint8)).save(out)
        print(f"[INFO] saved mask to {out}")


if __name__ == "__main__":
    main()
