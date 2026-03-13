"""Unit tests for dataloaders.base module."""

import tempfile
from pathlib import Path

import torch
from PIL import Image

from dataloaders.base import (
    BaseImageDataset,
    BaseSurgicalDataset,
    BaseVideoClipDataset,
)


class ConcreteImageDataset(BaseImageDataset):
    """Concrete implementation for testing."""

    def __init__(self, paths, transform=None):
        super().__init__(transform=transform)
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self._load_and_transform(self.paths[idx])


class TestBaseSurgicalDataset:
    """Test BaseSurgicalDataset name property."""

    def test_name_property(self):
        """name returns class name."""
        class DummyDataset(BaseSurgicalDataset):
            def __len__(self):
                return 0

            def __getitem__(self, idx):
                raise NotImplementedError

        ds = DummyDataset()
        assert ds.name == "DummyDataset"


class TestBaseImageDataset:
    """Test BaseImageDataset."""

    def test_transform_none(self):
        """Works when transform is None."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "img.png"
            Image.new("RGB", (8, 8)).save(p)
            ds = ConcreteImageDataset(paths=[str(p)], transform=None)
            assert ds.transform is None
            out = ds[0]
            assert out.shape == (3, 8, 8)

    def test_transform_set(self):
        """Accepts and applies transform callable."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "img.png"
            Image.new("RGB", (8, 8), color=(128, 128, 128)).save(p)
            t = lambda x: x * 2
            ds = ConcreteImageDataset(paths=[str(p)], transform=t)
            assert ds.transform is t
            out = ds[0]
            # 128/255 * 2 ≈ 1.0
            assert out.max() >= 1.0


class TestBaseVideoClipDataset:
    """Test BaseVideoClipDataset."""

    def test_default_num_frames(self):
        """Default num_frames is 16."""
        class Dummy(BaseVideoClipDataset):
            def __len__(self):
                return 0

            def __getitem__(self, idx):
                raise NotImplementedError

        ds = Dummy()
        assert ds.num_frames == 16

    def test_custom_num_frames(self):
        """Custom num_frames is stored."""
        class Dummy(BaseVideoClipDataset):
            def __len__(self):
                return 0

            def __getitem__(self, idx):
                raise NotImplementedError

        ds = Dummy(num_frames=32)
        assert ds.num_frames == 32
