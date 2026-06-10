# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""SAM / SAM2 cable segmentation wrapper (optional dependency)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SamCableSegmentorCfg:
    """Configuration for cable segmentation."""

    model_type: str = "vit_b"
    checkpoint_path: str = ""
    device: str = "cuda"


class SamCableSegmentor:
    """Thin wrapper around Segment Anything for cable-like masks."""

    def __init__(self, cfg: SamCableSegmentorCfg):
        self.cfg = cfg
        self._predictor = None

    def _lazy_init(self) -> None:
        if self._predictor is not None:
            return
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "Install perception extras: pip install -e 'source/unitree_rl_lab[perception]'"
            ) from exc

        if not self.cfg.checkpoint_path:
            raise ValueError("SamCableSegmentor requires checkpoint_path in cfg.")
        sam = sam_model_registry[self.cfg.model_type](checkpoint=self.cfg.checkpoint_path)
        sam.to(device=self.cfg.device)
        self._predictor = SamPredictor(sam)

    def segment(
        self,
        rgb: np.ndarray,
        point_prompts: np.ndarray | None = None,
        box_prompt: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return binary mask (H, W) uint8 for the cable."""
        self._lazy_init()
        assert self._predictor is not None
        self._predictor.set_image(rgb)
        masks: Any
        if box_prompt is not None:
            masks, _, _ = self._predictor.predict(box=box_prompt, multimask_output=False)
        elif point_prompts is not None:
            labels = np.ones(point_prompts.shape[0], dtype=np.int32)
            masks, _, _ = self._predictor.predict(
                point_coords=point_prompts,
                point_labels=labels,
                multimask_output=False,
            )
        else:
            h, w = rgb.shape[:2]
            center = np.array([[w * 0.5, h * 0.6]])
            labels = np.array([1])
            masks, _, _ = self._predictor.predict(
                point_coords=center,
                point_labels=labels,
                multimask_output=False,
            )
        mask = masks[0].astype(np.uint8)
        return (mask > 0).astype(np.uint8)
