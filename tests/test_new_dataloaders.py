"""Unit tests for CholecSeg8k, CholecTrack20, and CholecT50 dataloaders."""

from pathlib import Path

import pytest
import torch

from dataloaders.cholecseg8k import CholecSeg8kDataset, cholecseg8k_collate_fn
from dataloaders.cholectrack20 import (
    CholecTrack20DetectionDataset,
    cholectrack20_detection_collate_fn,
)
from dataloaders.cholect50 import (
    CholecT50PhaseDataset,
    CholecT50TripletDataset,
    cholect50_collate_fn,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class TestCholecSeg8k:
    def test_loads_dataset(self):
        ds = CholecSeg8kDataset(root=str(FIXTURES / "cholecseg8k"))
        assert len(ds) >= 1
        image, mask = ds[0]
        assert image.shape == (3, 480, 854)
        assert mask.shape == (480, 854)
        assert mask.dtype == torch.int64

    def test_collate_fn(self):
        ds = CholecSeg8kDataset(root=str(FIXTURES / "cholecseg8k"))
        batch = [ds[0], ds[0]]
        images, masks = cholecseg8k_collate_fn(batch)
        assert images.shape == (2, 3, 480, 854)
        assert masks.shape == (2, 480, 854)

    def test_raises_when_empty(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError, match="No CholecSeg8k images"):
                CholecSeg8kDataset(root=tmp)


class TestCholecTrack20:
    def test_loads_dataset(self):
        ds = CholecTrack20DetectionDataset(
            root=str(FIXTURES / "cholectrack20"),
            split="train",
        )
        assert len(ds) == 2
        image, target = ds[0]
        assert image.shape == (3, 480, 854)
        assert "boxes" in target
        assert "labels" in target
        assert target["video_id"] == "VID01"

    def test_collate_fn(self):
        ds = CholecTrack20DetectionDataset(root=str(FIXTURES / "cholectrack20"), split="train")
        images, targets = cholectrack20_detection_collate_fn([ds[0], ds[1]])
        assert len(images) == 2
        assert len(targets) == 2

    def test_raises_when_split_missing(self):
        with pytest.raises(FileNotFoundError, match="Split directory not found"):
            CholecTrack20DetectionDataset(root="/tmp", split="train")


class TestCholecT50:
    def test_triplet_dataset(self):
        ds = CholecT50TripletDataset(root=str(FIXTURES / "cholect50"))
        assert len(ds) == 2
        image, triplets, phase = ds[0]
        assert image.shape == (3, 480, 854)
        assert isinstance(triplets, list)
        assert len(triplets) >= 1
        assert "triplet_id" in triplets[0]
        assert phase >= 0

    def test_phase_dataset(self):
        ds = CholecT50PhaseDataset(root=str(FIXTURES / "cholect50"))
        assert len(ds) == 2
        image, phase = ds[0]
        assert phase >= 0

    def test_collate_fn(self):
        ds = CholecT50TripletDataset(root=str(FIXTURES / "cholect50"))
        images, triplets, phases = cholect50_collate_fn([ds[0], ds[1]])
        assert images.shape == (2, 3, 480, 854)
        assert len(triplets) == 2
        assert phases.shape == (2,)
