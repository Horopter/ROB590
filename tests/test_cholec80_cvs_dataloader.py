"""Unit tests for Cholec80-CVS dataloader."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from dataloaders.cholec80_cvs import (
    Cholec80CVSFrameDataset,
    Cholec80CVSTemporalDataset,
    _normalize_columns,
    cholec80_cvs_frame_collate_fn,
    cholec80_cvs_temporal_collate_fn,
)
from dataloaders.io import build_video_path, read_video_clip


# Path to fixtures
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_CSV = FIXTURES_DIR / "cholec80_cvs_sample.csv"
SAMPLE_TRAIN_CSV = FIXTURES_DIR / "cholec80_cvs_train_sample.csv"


class TestColumnNormalization:
    """Test column name normalization."""

    def test_normalize_standard_columns(self):
        df = pd.DataFrame(
            {
                "Video": [1],
                "Critical View": [0],
                "Initial Minute": [2],
                "Initial Second": [15],
                "Final Minute": [2],
                "Final Second": [25],
                "Two Structures": [1],
                "Cystic Plate": [0],
                "Hepatocystic Triangle": [0],
                "Total": [1],
            }
        )
        result = _normalize_columns(df)
        assert "video" in result.columns
        assert "critical_view" in result.columns
        assert "initial_minute" in result.columns
        assert "start_sec" not in result.columns  # computed later in dataset

    def test_normalize_preserves_values(self):
        df = pd.DataFrame({"Video": [1, 2], "Total": [3, 5]})
        result = _normalize_columns(df)
        assert list(result["video"]) == [1, 2]
        assert list(result["total"]) == [3, 5]


class TestVideoPath:
    """Test video path construction."""

    def test_video_path_default_pattern(self):
        path = build_video_path("/videos", 1, ".mp4", "video{:02d}")
        assert path == "/videos/video01.mp4"

    def test_video_path_video_80(self):
        path = build_video_path("/videos", 80, ".mp4", "video{:02d}")
        assert path == "/videos/video80.mp4"

    def test_video_path_string_id(self):
        path = build_video_path("/videos", "5", ".mp4", "video{:02d}")
        assert path == "/videos/video05.mp4"


class TestCholec80CVSTemporalDataset:
    """Test Cholec80-CVS temporal dataset."""

    @pytest.fixture
    def temp_video_dir(self):
        """Create temp dir with dummy video files."""
        with tempfile.TemporaryDirectory() as tmp:
            for i in [1, 2, 3]:
                # Create empty file as placeholder (real tests would need actual video)
                path = Path(tmp) / f"video{i:02d}.mp4"
                path.touch()
            yield tmp

    def test_dataset_requires_csv(self):
        """Dataset raises if required columns are missing."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"Video,Total\n1,3\n")
            f.flush()
            with pytest.raises(ValueError, match="Missing required columns"):
                Cholec80CVSTemporalDataset(
                    annotations_path=f.name,
                    video_root="/nonexistent",
                )

    def test_dataset_loads_csv(self):
        """Dataset loads valid CSV and computes length."""
        with tempfile.TemporaryDirectory() as video_dir:
            for i in [1, 2, 3]:
                (Path(video_dir) / f"video{i:02d}.mp4").touch()

            with patch(
                "dataloaders.base.read_video_clip",
                return_value=torch.rand(16, 3, 224, 224),
            ):
                ds = Cholec80CVSTemporalDataset(
                    annotations_path=str(SAMPLE_CSV),
                    video_root=video_dir,
                )
                assert len(ds) == 4

    def test_dataset_classification_task(self):
        """Classification task returns binary labels."""
        with tempfile.TemporaryDirectory() as video_dir:
            for i in [1, 2, 3]:
                (Path(video_dir) / f"video{i:02d}.mp4").touch()

            with patch(
                "dataloaders.base.read_video_clip",
                return_value=torch.rand(16, 3, 224, 224),
            ):
                ds = Cholec80CVSTemporalDataset(
                    annotations_path=str(SAMPLE_CSV),
                    video_root=video_dir,
                    task="classification",
                )
                clip, label, meta = ds[0]
                assert label.dim() == 0
                assert label.item() in (0, 1)
                assert meta["total"] == 1
                assert meta["critical_view"] is False

                # Row 2 (index 2) has Total=5 -> critical view
                clip, label, meta = ds[2]
                assert label.item() == 1
                assert meta["critical_view"] is True
                assert meta["total"] == 5

    def test_dataset_regression_task(self):
        """Regression task returns score vector."""
        with tempfile.TemporaryDirectory() as video_dir:
            for i in [1, 2, 3]:
                (Path(video_dir) / f"video{i:02d}.mp4").touch()

            with patch(
                "dataloaders.base.read_video_clip",
                return_value=torch.rand(16, 3, 224, 224),
            ):
                ds = Cholec80CVSTemporalDataset(
                    annotations_path=str(SAMPLE_CSV),
                    video_root=video_dir,
                    task="regression",
                )
                clip, label, meta = ds[0]
                assert label.shape == (4,)
                assert list(label.tolist()) == [1, 0, 0, 1]

    def test_dataset_metadata(self):
        """Metadata contains expected keys."""
        with tempfile.TemporaryDirectory() as video_dir:
            (Path(video_dir) / "video01.mp4").touch()

            with patch(
                "dataloaders.base.read_video_clip",
                return_value=torch.rand(16, 3, 224, 224),
            ):
                ds = Cholec80CVSTemporalDataset(
                    annotations_path=str(SAMPLE_CSV),
                    video_root=video_dir,
                )
                _, _, meta = ds[0]
                assert "video_id" in meta
                assert "video_path" in meta
                assert "start_sec" in meta
                assert "end_sec" in meta
                assert "two_structures" in meta
                assert "cystic_plate" in meta
                assert "hepatocystic_triangle" in meta
                assert "total" in meta
                assert meta["start_sec"] == 2 * 60 + 15  # 2 min 15 sec
                assert meta["end_sec"] == 2 * 60 + 25


class TestCholec80CVSCollateFn:
    """Test collate function."""

    def test_collate_classification_batch(self):
        """Collate stacks clips and labels for classification."""
        batch = [
            (torch.rand(16, 3, 224, 224), torch.tensor(0), {}),
            (torch.rand(16, 3, 224, 224), torch.tensor(1), {}),
        ]
        clips, labels, meta = cholec80_cvs_temporal_collate_fn(batch)
        assert clips.shape == (2, 16, 3, 224, 224)
        assert labels.shape == (2,)
        assert len(meta) == 2

    def test_collate_regression_batch(self):
        """Collate stacks clips and labels for regression."""
        batch = [
            (torch.rand(16, 3, 224, 224), torch.tensor([1.0, 0.0, 0.0, 1.0]), {}),
            (torch.rand(16, 3, 224, 224), torch.tensor([2.0, 2.0, 1.0, 5.0]), {}),
        ]
        clips, labels, meta = cholec80_cvs_temporal_collate_fn(batch)
        assert clips.shape == (2, 16, 3, 224, 224)
        assert labels.shape == (2, 4)


class TestDataLoaderIntegration:
    """Integration tests with DataLoader."""

    def test_dataloader_iteration(self):
        """DataLoader can iterate over dataset."""
        with tempfile.TemporaryDirectory() as video_dir:
            for i in [1, 2, 3]:
                (Path(video_dir) / f"video{i:02d}.mp4").touch()

            with patch(
                "dataloaders.base.read_video_clip",
                return_value=torch.rand(16, 3, 224, 224),
            ):
                ds = Cholec80CVSTemporalDataset(
                    annotations_path=str(SAMPLE_CSV),
                    video_root=video_dir,
                )
                loader = DataLoader(
                    ds,
                    batch_size=2,
                    collate_fn=cholec80_cvs_temporal_collate_fn,
                )
                clips, labels, meta = next(iter(loader))
                assert clips.shape[0] == 2
                assert labels.shape[0] == 2
                assert len(meta) == 2


class TestCholec80CVSFrameDataset:
    """Test Cholec80-CVS frame-level dataset (CHOLEC80-CVS-PUBLIC compatible)."""

    def test_dataset_loads_processed_csv(self):
        """Frame dataset loads from processed train/val/test CSV format."""
        with tempfile.TemporaryDirectory() as frames_root:
            # Create dummy frame structure: video01/100.jpg, video01/101.jpg, etc.
            for video, imgs in [("video01", [100, 101]), ("video02", [50, 51])]:
                (Path(frames_root) / video).mkdir()
                for i in imgs:
                    img = Image.new("RGB", (64, 64), color=(i % 256, 0, 0))
                    img.save(Path(frames_root) / video / f"{i}.jpg")

            ds = Cholec80CVSFrameDataset(
                set_name="train",
                frames_root=frames_root,
                data_dir=str(FIXTURES_DIR),
                predefined_set=pd.read_csv(SAMPLE_TRAIN_CSV),
            )
            assert len(ds) == 4

    def test_dataset_binarize_scores(self):
        """Binarize_scores clips scores to 0/1."""
        with tempfile.TemporaryDirectory() as frames_root:
            (Path(frames_root) / "video02").mkdir()
            Image.new("RGB", (64, 64)).save(Path(frames_root) / "video02" / "51.jpg")

            ds = Cholec80CVSFrameDataset(
                set_name="train",
                frames_root=frames_root,
                data_dir=str(FIXTURES_DIR),
                predefined_set=pd.read_csv(SAMPLE_TRAIN_CSV),
                binarize_scores=True,
            )
            _, target, _ = ds[3]  # Row with 2,2,1
            assert list(target.tolist()) == [1, 1, 1]  # min(1, x)

    def test_frame_collate_fn(self):
        """Frame collate stacks images and targets."""
        batch = [
            (torch.rand(3, 64, 64), torch.tensor([1.0, 0.0, 0.0]), "video01/0.jpg"),
            (torch.rand(3, 64, 64), torch.tensor([0.0, 1.0, 0.0]), "video02/1.jpg"),
        ]
        images, targets, paths = cholec80_cvs_frame_collate_fn(batch)
        assert images.shape == (2, 3, 64, 64)
        assert targets.shape == (2, 3)
        assert paths == ["video01/0.jpg", "video02/1.jpg"]


class TestReadVideoClipOpencv:
    """Test video reading (requires opencv)."""

    def test_raises_for_nonexistent_video(self):
        """Raises FileNotFoundError when video file does not exist."""
        try:
            import cv2
        except ImportError:
            pytest.skip("opencv-python not installed")
        with pytest.raises(FileNotFoundError, match="Could not open video"):
            read_video_clip("/nonexistent/video.mp4", 0, 1, 16)


class TestCholec80CVSXlsxAndOptions:
    """Test XLSX loading and skip_invalid_segments."""

    def test_dataset_loads_xlsx(self):
        """Dataset loads from XLSX when openpyxl available."""
        try:
            import openpyxl
        except ImportError:
            pytest.skip("openpyxl not installed")
        with tempfile.TemporaryDirectory() as tmp:
            video_dir = Path(tmp) / "videos"
            video_dir.mkdir()
            for i in [1, 2, 3]:
                (video_dir / f"video{i:02d}.mp4").touch()
            df = pd.read_csv(SAMPLE_CSV)
            xlsx_path = Path(tmp) / "ann.xlsx"
            df.to_excel(xlsx_path, index=False, engine="openpyxl")
            with patch(
                "dataloaders.base.read_video_clip",
                return_value=torch.rand(16, 3, 224, 224),
            ):
                ds = Cholec80CVSTemporalDataset(
                    annotations_path=str(xlsx_path),
                    video_root=str(video_dir),
                )
                assert len(ds) == 4

    def test_skip_invalid_segments(self):
        """skip_invalid_segments=False keeps rows with start_sec >= end_sec."""
        with tempfile.TemporaryDirectory() as video_dir:
            (Path(video_dir) / "video01.mp4").touch()
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                # Row with start >= end
                f.write(
                    b"Video,Critical View,Initial Minute,Initial Second,Final Minute,Final Second,"
                    b"Two Structures,Cystic Plate,Hepatocystic Triangle,Total\n"
                    b"1,0,2,30,2,25,1,0,0,1\n"
                )
                f.flush()
                try:
                    with patch(
                        "dataloaders.base.read_video_clip",
                        return_value=torch.rand(16, 3, 224, 224),
                    ):
                        ds_skip = Cholec80CVSTemporalDataset(
                            annotations_path=f.name,
                            video_root=video_dir,
                            skip_invalid_segments=True,
                        )
                        ds_no_skip = Cholec80CVSTemporalDataset(
                            annotations_path=f.name,
                            video_root=video_dir,
                            skip_invalid_segments=False,
                        )
                    assert len(ds_skip) == 0
                    assert len(ds_no_skip) == 1
                finally:
                    os.unlink(f.name)
