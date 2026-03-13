"""Unit tests for dataloaders.io module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from dataloaders.io import build_video_path, load_image, read_video_clip


class TestLoadImage:
    """Test load_image function."""

    def test_load_image_as_tensor(self):
        """Returns [C, H, W] tensor in [0, 1] when as_tensor=True."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (32, 32), color=(128, 64, 192))
            img.save(f.name)
            try:
                out = load_image(f.name, as_tensor=True)
                assert isinstance(out, torch.Tensor)
                assert out.shape == (3, 32, 32)
                assert out.dtype == torch.float32
                assert 0 <= out.min().item() <= 1
                assert 0 <= out.max().item() <= 1
            finally:
                os.unlink(f.name)

    def test_load_image_as_numpy(self):
        """Returns [H, W, C] numpy array when as_tensor=False."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (16, 16), color=(255, 0, 0))
            img.save(f.name)
            try:
                out = load_image(f.name, as_tensor=False)
                assert isinstance(out, np.ndarray)
                assert out.shape == (16, 16, 3)
                assert out.dtype == np.uint8 or out.dtype == np.int64
            finally:
                os.unlink(f.name)


class TestBuildVideoPath:
    """Test build_video_path function."""

    def test_build_video_path_int_id(self):
        """Integer video ID uses pattern."""
        path = build_video_path("/root", 7, ".mp4", "video{:02d}")
        assert path == "/root/video07.mp4"

    def test_build_video_path_string_digit_id(self):
        """String digit ID is converted to int for pattern."""
        path = build_video_path("/root", "12", ".avi", "video{:02d}")
        assert path == "/root/video12.avi"

    def test_build_video_path_non_numeric_string(self):
        """Non-numeric string ID used as-is."""
        path = build_video_path("/root", "custom_video", ".mp4", "video{:02d}")
        assert path == "/root/custom_video.mp4"


class TestReadVideoClip:
    """Test read_video_clip function."""

    def test_raises_when_cv2_unavailable(self):
        """Raises ImportError when opencv is not installed."""
        with patch("dataloaders.io.CV2_AVAILABLE", False):
            with pytest.raises(ImportError, match="opencv-python"):
                read_video_clip("/any/path.mp4", 0, 1, 16)
