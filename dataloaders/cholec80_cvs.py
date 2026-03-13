"""
Cholec80-CVS Dataset Loader

Loads Strasberg's Critical View of Safety (CVS) annotations.
Compatible with CHOLEC80-CVS-PUBLIC: https://github.com/ManuelRios18/CHOLEC80-CVS-PUBLIC
"""

import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from .base import BaseImageDataset, BaseTemporalActionDataset
from .io import build_video_path, load_image

COLUMN_ALIASES = {
    "video": ["Video", "video", "VIDEO"],
    "critical_view": ["Critical View", "critical view", "CriticalView", "critical_view"],
    "initial_minute": ["Initial Minute", "initial minute", "InitialMinute", "initial_minute"],
    "initial_second": ["Initial Second", "initial second", "InitialSecond", "initial_second"],
    "final_minute": ["Final Minute", "final minute", "FinalMinute", "final_minute"],
    "final_second": ["Final Second", "final second", "FinalSecond", "final_second"],
    "two_structures": ["Two Structures", "two structures", "TwoStructures", "two_structures"],
    "cystic_plate": ["Cystic Plate", "cystic plate", "CysticPlate", "cystic_plate"],
    "hepatocystic_triangle": ["Hepatocystic Triangle", "hepatocystic triangle", "HepatocysticTriangle", "hepatocystic_triangle"],
    "total": ["Total", "total", "TOTAL"],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename[alias] = canonical
                break
    return df.rename(columns=rename)


class Cholec80CVSTemporalDataset(BaseTemporalActionDataset):
    """Temporal segment annotations for video clip loading."""

    def __init__(
        self,
        annotations_path: str,
        video_root: str,
        video_ext: str = ".mp4",
        video_name_pattern: str = "video{:02d}",
        num_frames: int = 16,
        transform: Optional[Callable] = None,
        task: str = "classification",
        cvs_threshold: int = 5,
        skip_invalid_segments: bool = True,
    ):
        super().__init__(num_frames=num_frames, transform=transform)
        self.video_root = video_root
        self.video_ext = video_ext
        self.video_name_pattern = video_name_pattern
        self.task = task
        self.cvs_threshold = cvs_threshold

        if annotations_path.lower().endswith(".xlsx"):
            self.df = pd.read_excel(annotations_path, engine="openpyxl")
        else:
            self.df = pd.read_csv(annotations_path)
        self.df = _normalize_columns(self.df)

        required = {"video", "initial_minute", "initial_second", "final_minute", "final_second",
                    "two_structures", "cystic_plate", "hepatocystic_triangle", "total"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Found: {list(self.df.columns)}")

        self.df["start_sec"] = self.df["initial_minute"] * 60 + self.df["initial_second"]
        self.df["end_sec"] = self.df["final_minute"] * 60 + self.df["final_second"]
        if skip_invalid_segments:
            self.df = self.df[self.df["start_sec"] < self.df["end_sec"]].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _video_path(self, idx: int) -> str:
        row = self.df.iloc[idx]
        return build_video_path(
            self.video_root, row["video"],
            self.video_ext, self.video_name_pattern,
        )

    def _start_sec(self, idx: int) -> float:
        return float(self.df.iloc[idx]["start_sec"])

    def _end_sec(self, idx: int) -> float:
        return float(self.df.iloc[idx]["end_sec"])

    def _get_label(self, idx: int) -> torch.Tensor:
        row = self.df.iloc[idx]
        total = int(row["total"])
        if self.task == "classification":
            return torch.tensor(1 if total >= self.cvs_threshold else 0, dtype=torch.long)
        return torch.tensor(
            [int(row["two_structures"]), int(row["cystic_plate"]),
             int(row["hepatocystic_triangle"]), total],
            dtype=torch.float32,
        )

    def _get_metadata(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        total = int(row["total"])
        return {
            "video_id": row["video"],
            "video_path": self._video_path(idx),
            "start_sec": float(row["start_sec"]),
            "end_sec": float(row["end_sec"]),
            "two_structures": int(row["two_structures"]),
            "cystic_plate": int(row["cystic_plate"]),
            "hepatocystic_triangle": int(row["hepatocystic_triangle"]),
            "total": total,
            "critical_view": total >= self.cvs_threshold,
        }


class Cholec80CVSFrameDataset(BaseImageDataset):
    """Frame-level dataset (CHOLEC80-CVS-PUBLIC pipeline output)."""

    def __init__(
        self,
        set_name: str,
        frames_root: str,
        data_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        binarize_scores: bool = True,
        predefined_set: Optional[pd.DataFrame] = None,
    ):
        super().__init__(transform=transform)
        assert set_name in ["train", "val", "test"]
        self.frames_root = frames_root
        self.binarize_scores = binarize_scores

        if predefined_set is not None:
            self.labels_df = predefined_set
        else:
            data_dir = data_dir or "data"
            self.labels_df = pd.read_csv(os.path.join(data_dir, f"{set_name}.csv"))

        required = {"video_name", "image", "two_structures_score", "cystic_plate_score", "hc_triangle_score"}
        if required - set(self.labels_df.columns):
            raise ValueError(f"Missing required columns. Run CHOLEC80-CVS-PUBLIC pipeline first.")

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.labels_df.iloc[index]
        path = os.path.join(self.frames_root, row["video_name"], f"{int(row['image'])}.jpg")
        image = self._load_and_transform(path)

        if self.binarize_scores:
            scores = [int(min(1, row["two_structures_score"])),
                      int(min(1, row["cystic_plate_score"])),
                      int(min(1, row["hc_triangle_score"]))]
        else:
            scores = [int(row["two_structures_score"]), int(row["cystic_plate_score"]), int(row["hc_triangle_score"])]
        target = torch.tensor(scores, dtype=torch.float32)
        return image, target, f"{row['video_name']}/{int(row['image'])}.jpg"


def cholec80_cvs_temporal_collate_fn(batch):
    clips, labels, metadata = zip(*batch)
    return torch.stack(clips, dim=0), torch.stack(labels, dim=0), list(metadata)


def cholec80_cvs_frame_collate_fn(batch):
    images, targets, paths = zip(*batch)
    return torch.stack(images, dim=0), torch.stack(targets, dim=0), list(paths)
