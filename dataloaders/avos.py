"""
AVOS Dataset Loader

Object detection and temporal action recognition.
"""

import os
import ast
from typing import Any, Dict, Optional

import pandas as pd
import torch

from .base import BaseDetectionDataset, BaseTemporalActionDataset
from .io import load_image, read_video_clip

BBOX_LABEL_MAP = {"hand": 1, "bovie": 2, "needledriver": 3, "forceps": 4}
TEMPORAL_LABEL_MAP = {"background": 0, "cutting": 1, "suturing": 2, "tying": 3}


def _extract_video_id(image_name: str) -> str:
    base = os.path.basename(image_name)
    return os.path.splitext(base)[0].rsplit("-", 1)[0]


def _safe_literal_eval(x):
    if not isinstance(x, str):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return x


class AVOSBoundingBoxDataset(BaseDetectionDataset):
    """Loads AVOS object detection annotations."""

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        transforms=None,
        label_map: Optional[Dict[str, int]] = None,
        return_video_id: bool = True,
    ):
        super().__init__(transform=transforms)
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.label_map = label_map or BBOX_LABEL_MAP
        self.return_video_id = return_video_id
        required = {"image_name", "x1", "y1", "x2", "y2", "class"}
        if required - set(self.df.columns):
            raise ValueError(f"Missing required columns: {required - set(self.df.columns)}")
        self.grouped = list(self.df.groupby("image_name"))

    def __len__(self) -> int:
        return len(self.grouped)

    def _image_path(self, idx: int) -> str:
        image_name, _ = self.grouped[idx]
        return os.path.join(self.image_root, image_name)

    def _build_target(self, idx: int) -> Dict[str, Any]:
        _, group = self.grouped[idx]
        boxes, labels = [], []
        for _, row in group.iterrows():
            cls = str(row["class"]).strip().lower()
            if cls not in self.label_map:
                continue
            boxes.append([float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])])
            labels.append(self.label_map[cls])
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) > 0 else torch.zeros((0,), dtype=torch.float32)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
            "image_name": self.grouped[idx][0],
        }
        if self.return_video_id:
            target["video_id"] = _extract_video_id(self.grouped[idx][0])
        return target


class AVOSTemporalActionDataset(BaseTemporalActionDataset):
    """Loads AVOS temporal action annotations."""

    def __init__(
        self,
        csv_path: str,
        video_root: str,
        video_ext: str = ".mp4",
        num_frames: int = 16,
        transform=None,
        label_map: Optional[Dict[str, int]] = None,
    ):
        super().__init__(num_frames=num_frames, transform=transform)
        self.df = pd.read_csv(csv_path)
        self.video_root = video_root
        self.video_ext = video_ext
        self.label_map = label_map or TEMPORAL_LABEL_MAP
        required = {"video_id", "start_seconds", "end_seconds", "start_frame", "end_frame", "label"}
        if required - set(self.df.columns):
            raise ValueError(f"Missing required columns: {required - set(self.df.columns)}")
        self.df = self.df.drop(columns=[c for c in self.df.columns if str(c).startswith("Unnamed:")], errors="ignore")

    def __len__(self) -> int:
        return len(self.df)

    def _video_path(self, idx: int) -> str:
        return os.path.join(self.video_root, f"{self.df.iloc[idx]['video_id']}{self.video_ext}")

    def _start_sec(self, idx: int) -> float:
        return float(self.df.iloc[idx]["start_seconds"])

    def _end_sec(self, idx: int) -> float:
        return float(self.df.iloc[idx]["end_seconds"])

    def _get_label(self, idx: int) -> torch.Tensor:
        label_name = str(self.df.iloc[idx]["label"]).strip().lower()
        if label_name not in self.label_map:
            raise KeyError(f"Unknown temporal label: {label_name}")
        return torch.tensor(self.label_map[label_name], dtype=torch.long)

    def _get_metadata(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        return {
            "video_id": str(row["video_id"]),
            "video_path": self._video_path(idx),
            "start_seconds": float(row["start_seconds"]),
            "end_seconds": float(row["end_seconds"]),
            "start_frame": int(row["start_frame"]),
            "end_frame": int(row["end_frame"]),
            "label_name": str(row["label"]).strip().lower(),
        }


def detection_collate_fn(batch):
    if len(batch[0]) == 2:
        images, targets = zip(*batch)
        return list(images), list(targets)
    images, targets, metadata = zip(*batch)
    return list(images), list(targets), list(metadata)


def temporal_collate_fn(batch):
    clips, labels, metadata = zip(*batch)
    return torch.stack(clips, dim=0), torch.stack(labels, dim=0), list(metadata)
