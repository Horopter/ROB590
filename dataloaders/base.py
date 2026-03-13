"""
Base classes for surgical video datasets.

Provides a consistent OOP hierarchy:
  BaseSurgicalDataset (ABC)
    ├── BaseImageDataset      (frame-level, single image)
    │     ├── BaseSegmentationDataset
    │     └── BaseDetectionDataset
    └── BaseVideoClipDataset  (clip-level, temporal)
          └── BaseTemporalActionDataset
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from .io import load_image, read_video_clip


class BaseSurgicalDataset(Dataset, ABC):
    """Abstract base for all surgical datasets."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        pass

    @property
    def name(self) -> str:
        """Dataset name for logging."""
        return self.__class__.__name__


class BaseImageDataset(BaseSurgicalDataset):
    """Base for frame-level datasets (single image per sample)."""

    def __init__(self, transform: Optional[Callable] = None):
        self.transform = transform

    def _load_and_transform(self, path: str) -> torch.Tensor:
        """Load image and apply transform."""
        image = load_image(path, as_tensor=True)
        if self.transform is not None:
            image = self.transform(image)
        return image


class BaseSegmentationDataset(BaseImageDataset):
    """Base for semantic segmentation (image, mask)."""

    def __init__(
        self,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(transform=transform)
        self.target_transform = target_transform

    @abstractmethod
    def _load_mask(self, idx: int) -> torch.Tensor:
        """Load segmentation mask for sample idx."""
        pass

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self._load_and_transform(self._image_path(idx))
        mask = self._load_mask(idx)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return image, mask

    @abstractmethod
    def _image_path(self, idx: int) -> str:
        pass


class BaseDetectionDataset(BaseImageDataset):
    """Base for object detection (image, target dict with boxes/labels)."""

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        image = self._load_and_transform(self._image_path(idx))
        target = self._build_target(idx)
        return image, target

    @abstractmethod
    def _image_path(self, idx: int) -> str:
        pass

    @abstractmethod
    def _build_target(self, idx: int) -> Dict[str, Any]:
        pass


class BaseVideoClipDataset(BaseSurgicalDataset):
    """Base for clip-level datasets (video segment per sample)."""

    def __init__(
        self,
        num_frames: int = 16,
        transform: Optional[Callable] = None,
    ):
        self.num_frames = num_frames
        self.transform = transform

    def _load_clip(self, video_path: str, start_sec: float, end_sec: float) -> torch.Tensor:
        """Load video clip as tensor [T, C, H, W]."""
        clip = read_video_clip(
            video_path=video_path,
            start_sec=start_sec,
            end_sec=end_sec,
            num_frames=self.num_frames,
        )
        if self.transform is not None:
            clip = self.transform(clip)
        return clip


class BaseTemporalActionDataset(BaseVideoClipDataset):
    """Base for temporal action recognition (clip, label, metadata)."""

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        clip = self._load_clip(
            self._video_path(idx),
            self._start_sec(idx),
            self._end_sec(idx),
        )
        label = self._get_label(idx)
        metadata = self._get_metadata(idx)
        return clip, label, metadata

    @abstractmethod
    def _video_path(self, idx: int) -> str:
        pass

    @abstractmethod
    def _start_sec(self, idx: int) -> float:
        pass

    @abstractmethod
    def _end_sec(self, idx: int) -> float:
        pass

    @abstractmethod
    def _get_label(self, idx: int) -> torch.Tensor:
        pass

    @abstractmethod
    def _get_metadata(self, idx: int) -> Dict[str, Any]:
        pass
