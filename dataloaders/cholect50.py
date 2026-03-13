"""
CholecT50 Dataset Loader

Surgical action triplet dataset: <instrument, verb, target>. 50 videos, ~100 triplet categories.
https://github.com/CAMMA-public/cholect50
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

from .base import BaseImageDataset


class CholecT50TripletDataset(BaseImageDataset):
    """Action triplet recognition dataset for CholecT50."""

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        return_bbox: bool = False,
        video_ids: Optional[List[str]] = None,
    ):
        super().__init__(transform=transform)
        self.root = Path(root)
        self.return_bbox = return_bbox

        videos_dir = self.root / "videos"
        labels_dir = self.root / "labels"
        if not videos_dir.exists() or not labels_dir.exists():
            raise FileNotFoundError(f"CholecT50 structure not found. Expected {self.root}/videos and {self.root}/labels")

        video_ids = video_ids or sorted([d.name for d in videos_dir.iterdir() if d.is_dir()])
        self.samples = []
        self.categories = {}
        for vid in video_ids:
            vid_dir = videos_dir / vid
            label_path = labels_dir / f"{vid}.json"
            if not vid_dir.exists() or not label_path.exists():
                continue
            data = json.load(open(label_path))
            if not self.categories:
                self.categories = data.get("categories", {})
            anns = data.get("annotations", {})
            for frame_id in sorted(anns.keys(), key=int):
                img_path = vid_dir / f"{int(frame_id):06d}.png"
                if img_path.exists():
                    self.samples.append((str(img_path), vid, int(frame_id), anns[frame_id], data))

    def __len__(self) -> int:
        return len(self.samples)

    def _parse_instance(self, inst: dict) -> dict:
        out = {
            "triplet_id": inst.get("ID", inst.get("id", inst.get("triplet_id", -1))),
            "instrument_id": inst.get("instrument", inst.get("instrument_id", -1)),
            "verb_id": inst.get("verb", inst.get("verb_id", -1)),
            "target_id": inst.get("target", inst.get("target_id", -1)),
        }
        if self.return_bbox:
            out["bbox"] = [inst.get("BX", -1), inst.get("BY", -1), inst.get("BW", -1), inst.get("BH", -1)]
        return out

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[dict], int]:
        img_path, video_id, frame_id, instances, data = self.samples[idx]
        image = self._load_and_transform(img_path)
        triplets = []
        phase = -1
        for inst in instances:
            if isinstance(inst, dict):
                triplets.append(self._parse_instance(inst))
                phase = inst.get("phase", phase)
        return image, triplets, phase


class CholecT50PhaseDataset(CholecT50TripletDataset):
    """Phase recognition only (frame -> phase label)."""

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, _, phase = super().__getitem__(idx)
        return image, phase


def cholect50_collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    triplets = [b[1] for b in batch]
    phases = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return images, triplets, phases
