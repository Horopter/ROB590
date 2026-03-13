"""
CholecTrack20 Dataset Loader

Multi-perspective surgical tool tracking. 20 videos, 35K+ frames, 7 tool categories.
https://github.com/camma-public/cholectrack20
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

from .base import BaseDetectionDataset
from .io import load_image

CHOLEcTRACK20_CATEGORIES = [
    "grasper", "bipolar", "hook", "scissors", "clipper", "irrigator", "specimen_bag",
]


class CholecTrack20DetectionDataset(BaseDetectionDataset):
    """Detection dataset for CholecTrack20 (bounding boxes)."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        return_metadata: bool = True,
    ):
        super().__init__(transform=transform)
        self.root = Path(root)
        self.split = split
        self.return_metadata = return_metadata

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.samples = []
        for vid_dir in sorted(split_dir.iterdir()):
            if not vid_dir.is_dir():
                continue
            json_path = vid_dir / f"{vid_dir.name}.json"
            images_dir = vid_dir / "images"
            if not json_path.exists() or not images_dir.exists():
                continue
            anns = json.load(open(json_path))
            records = anns.get("annotations", anns)
            for frame_id in sorted(records.keys(), key=int):
                img_path = images_dir / f"{int(frame_id):06d}.png"
                if img_path.exists():
                    self.samples.append((str(img_path), vid_dir.name, frame_id, records[frame_id]))

    def __len__(self) -> int:
        return len(self.samples)

    def _image_path(self, idx: int) -> str:
        return self.samples[idx][0]

    def _build_target(self, idx: int) -> Dict:
        _, video_id, frame_id, instances = self.samples[idx]
        boxes, labels, metadata_list = [], [], []
        for inst in instances:
            bbox = inst.get("tool_bbox", inst.get("bbox", []))
            if len(bbox) >= 4:
                x, y, w, h = bbox[:4]
                boxes.append([x, y, x + w, y + h])
                labels.append(inst.get("instrument", inst.get("category", 0)))
                if self.return_metadata:
                    metadata_list.append({
                        "operator": inst.get("operator", -1),
                        "phase": inst.get("phase", -1),
                        "intraoperative_track_id": inst.get("intraoperative_track_id", -1),
                    })

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.long)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "num_instances": len(boxes_t),
            "video_id": video_id,
            "frame_id": int(frame_id),
        }
        if self.return_metadata and metadata_list:
            target["instance_metadata"] = metadata_list
        return target


def cholectrack20_detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
