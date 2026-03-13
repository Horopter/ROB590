"""Unit tests for AVOS dataloaders."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from dataloaders.avos import (
    AVOSBoundingBoxDataset,
    AVOSTemporalActionDataset,
    BBOX_LABEL_MAP,
    TEMPORAL_LABEL_MAP,
    detection_collate_fn,
    temporal_collate_fn,
)


def _make_avos_bbox_csv(path: str) -> None:
    """Create minimal AVOS bbox CSV."""
    df = pd.DataFrame({
        "image_name": ["vid1-frame001.jpg", "vid1-frame001.jpg", "vid1-frame002.jpg"],
        "x1": [10, 50, 20],
        "y1": [20, 60, 30],
        "x2": [30, 70, 40],
        "y2": [40, 80, 50],
        "class": ["hand", "bovie", "forceps"],
    })
    df.to_csv(path, index=False)


def _make_avos_temporal_csv(path: str) -> None:
    """Create minimal AVOS temporal CSV."""
    df = pd.DataFrame({
        "video_id": ["v1", "v1"],
        "start_seconds": [0.0, 5.0],
        "end_seconds": [2.0, 7.0],
        "start_frame": [0, 125],
        "end_frame": [50, 175],
        "label": ["cutting", "suturing"],
    })
    df.to_csv(path, index=False)


class TestAVOSBoundingBoxDataset:
    """Test AVOSBoundingBoxDataset."""

    def test_raises_missing_columns(self):
        """Raises when required columns are missing."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            pd.DataFrame({"image_name": ["a.jpg"], "x1": [0]}).to_csv(f.name, index=False)
            try:
                with pytest.raises(ValueError, match="Missing required columns"):
                    AVOSBoundingBoxDataset(
                        csv_path=f.name,
                        image_root="/tmp",
                    )
            finally:
                os.unlink(f.name)

    def test_loads_and_builds_target(self):
        """Loads CSV and builds detection target."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "ann.csv"
            img_root = Path(tmp) / "images"
            img_root.mkdir()
            _make_avos_bbox_csv(str(csv_path))
            for name in ["vid1-frame001.jpg", "vid1-frame002.jpg"]:
                Image.new("RGB", (64, 64)).save(img_root / name)

            ds = AVOSBoundingBoxDataset(csv_path=str(csv_path), image_root=str(img_root))
            assert len(ds) == 2
            image, target = ds[0]
            assert image.shape == (3, 64, 64)
            assert "boxes" in target
            assert "labels" in target
            assert target["boxes"].shape[0] == 2
            assert target["labels"].shape[0] == 2
            assert "video_id" in target
            assert target["video_id"] == "vid1"

    def test_custom_label_map(self):
        """Uses custom label_map when provided."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "ann.csv"
            img_root = Path(tmp) / "images"
            img_root.mkdir()
            df = pd.DataFrame({
                "image_name": ["a.jpg"],
                "x1": [0], "y1": [0], "x2": [10], "y2": [10],
                "class": ["hand"],
            })
            df.to_csv(csv_path, index=False)
            Image.new("RGB", (64, 64)).save(img_root / "a.jpg")

            custom_map = {"hand": 10}
            ds = AVOSBoundingBoxDataset(
                csv_path=str(csv_path),
                image_root=str(img_root),
                label_map=custom_map,
            )
            _, target = ds[0]
            assert target["labels"][0].item() == 10

    def test_return_video_id_false(self):
        """Omits video_id when return_video_id=False."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "ann.csv"
            img_root = Path(tmp) / "images"
            img_root.mkdir()
            df = pd.DataFrame({
                "image_name": ["a.jpg"],
                "x1": [0], "y1": [0], "x2": [10], "y2": [10],
                "class": ["hand"],
            })
            df.to_csv(csv_path, index=False)
            Image.new("RGB", (64, 64)).save(img_root / "a.jpg")

            ds = AVOSBoundingBoxDataset(
                csv_path=str(csv_path),
                image_root=str(img_root),
                return_video_id=False,
            )
            _, target = ds[0]
            assert "video_id" not in target


class TestAVOSTemporalActionDataset:
    """Test AVOSTemporalActionDataset."""

    def test_raises_missing_columns(self):
        """Raises when required columns are missing."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            pd.DataFrame({"video_id": ["v1"], "label": ["cutting"]}).to_csv(f.name, index=False)
            try:
                with pytest.raises(ValueError, match="Missing required columns"):
                    AVOSTemporalActionDataset(
                        csv_path=f.name,
                        video_root="/tmp",
                    )
            finally:
                os.unlink(f.name)

    def test_raises_unknown_label(self):
        """Raises KeyError for unknown label."""
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "ann.csv"
            video_root = Path(tmp) / "videos"
            video_root.mkdir()
            (video_root / "v1.mp4").touch()
            df = pd.DataFrame({
                "video_id": ["v1"],
                "start_seconds": [0.0],
                "end_seconds": [2.0],
                "start_frame": [0],
                "end_frame": [50],
                "label": ["unknown_action"],
            })
            df.to_csv(csv_path, index=False)

            with patch(
                "dataloaders.base.read_video_clip",
                return_value=torch.rand(16, 3, 224, 224),
            ):
                with pytest.raises(KeyError, match="Unknown temporal label"):
                    ds = AVOSTemporalActionDataset(
                        csv_path=str(csv_path),
                        video_root=str(video_root),
                    )
                    _ = ds[0]


class TestAVOSCollateFn:
    """Test AVOS collate functions."""

    def test_detection_collate_two_items(self):
        """detection_collate_fn for (image, target) batches."""
        batch = [
            (torch.rand(3, 64, 64), {"boxes": torch.rand(2, 4), "labels": torch.tensor([1, 2])}),
            (torch.rand(3, 64, 64), {"boxes": torch.rand(1, 4), "labels": torch.tensor([3])}),
        ]
        images, targets = detection_collate_fn(batch)
        assert len(images) == 2
        assert len(targets) == 2

    def test_temporal_collate_fn(self):
        """temporal_collate_fn stacks clips and labels."""
        batch = [
            (torch.rand(16, 3, 224, 224), torch.tensor(0), {"video_id": "v1"}),
            (torch.rand(16, 3, 224, 224), torch.tensor(1), {"video_id": "v2"}),
        ]
        clips, labels, meta = temporal_collate_fn(batch)
        assert clips.shape == (2, 16, 3, 224, 224)
        assert labels.shape == (2,)
        assert len(meta) == 2
