"""
CholecSeg8k Dataset Loader

Semantic segmentation dataset for laparoscopic cholecystectomy.
8,080 frames from 17 Cholec80 videos, 13 classes.
https://www.kaggle.com/datasets/newslab/cholecseg8k
https://arxiv.org/abs/2012.12453
"""

import os
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .base import BaseSegmentationDataset


# 13 classes from CholecSeg8k paper (Table I)
CHOLEcSEG8K_CLASSES = [
    "black_background", "abdominal_wall", "liver", "gastrointestinal_tract",
    "fat", "grasper", "connective_tissue", "blood", "cystic_duct",
    "l_hook_electrocautery", "gallbladder", "hepatic_vein", "liver_ligament",
]
NUM_CLASSES = 13


def _find_image_files(root: str) -> List[Tuple[str, str, str]]:
    """Find all (image_path, mask_path, video_clip) tuples."""
    results = []
    for video_folder in sorted(os.listdir(root)):
        video_path = os.path.join(root, video_folder)
        if not os.path.isdir(video_path):
            continue
        for clip_folder in sorted(os.listdir(video_path)):
            clip_path = os.path.join(video_path, clip_folder)
            if not os.path.isdir(clip_path):
                continue
            for f in sorted(os.listdir(clip_path)):
                if not f.endswith(".png") or "mask" in f or "watershed" in f or "color" in f:
                    continue
                img_path = os.path.join(clip_path, f)
                base = f.replace(".png", "").replace("_endo", "")
                mask_path = None
                for suffix in ["_endo_watershed_mask.png", "_watershed_mask.png", "_endo_mask.png", "_mask.png"]:
                    p = os.path.join(clip_path, f"{base}{suffix}")
                    if os.path.exists(p):
                        mask_path = p
                        break
                if mask_path:
                    results.append((img_path, mask_path, f"{video_folder}/{clip_folder}"))
    return results


class CholecSeg8kDataset(BaseSegmentationDataset):
    """Semantic segmentation dataset for CholecSeg8k."""

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(transform=transform, target_transform=target_transform)
        root = os.path.join(root, "CholecSeg8k") if os.path.exists(os.path.join(root, "CholecSeg8k")) else root
        self.samples = _find_image_files(root)
        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No CholecSeg8k images found in {root}. "
                "Expected structure: root/videoXX/videoXX_startFrame/*_endo.png"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _image_path(self, idx: int) -> str:
        return self.samples[idx][0]

    def _load_mask(self, idx: int) -> torch.Tensor:
        mask_path = self.samples[idx][1]
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return torch.from_numpy(mask).long()


def cholecseg8k_collate_fn(batch):
    """Collate for semantic segmentation."""
    images, masks = zip(*batch)
    return torch.stack(images, dim=0), torch.stack(masks, dim=0)
