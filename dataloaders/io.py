"""
Shared I/O utilities for surgical video datasets.
"""

import os
from typing import Union

import numpy as np
import torch
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def load_image(path: str, as_tensor: bool = True) -> Union[np.ndarray, torch.Tensor]:
    """Load RGB image from path.

    Args:
        path: Image file path.
        as_tensor: If True, return [C, H, W] tensor in [0, 1]. Else numpy [H, W, C].

    Returns:
        Image as tensor or numpy array.
    """
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    if as_tensor:
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return arr


def read_video_clip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    num_frames: int = 16,
    fps: float = 25.0,
) -> torch.Tensor:
    """Load a clip from a video using OpenCV.

    Returns tensor of shape [T, C, H, W] in [0, 1].

    Args:
        video_path: Path to video file.
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
        num_frames: Number of frames to sample.
        fps: Fallback FPS if video metadata unavailable.

    Raises:
        ImportError: If opencv-python not installed.
        FileNotFoundError: If video cannot be opened.
    """
    if not CV2_AVAILABLE:
        raise ImportError("opencv-python is required. Install with: pip install opencv-python")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    if vid_fps is None or vid_fps <= 0:
        vid_fps = fps

    start_frame = int(start_sec * vid_fps)
    end_frame = int(end_sec * vid_fps)
    if end_frame <= start_frame:
        end_frame = start_frame + 1

    frame_indices = torch.linspace(start_frame, end_frame - 1, steps=num_frames).long().tolist()
    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frames.append(frame_tensor)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(
            f"No frames could be read from {video_path} between {start_sec} and {end_sec}"
        )

    while len(frames) < num_frames:
        frames.append(frames[-1].clone())

    return torch.stack(frames, dim=0)


def build_video_path(
    video_root: str,
    video_id: Union[int, str],
    video_ext: str = ".mp4",
    video_name_pattern: str = "video{:02d}",
) -> str:
    """Build video file path from video ID."""
    if isinstance(video_id, str) and video_id.isdigit():
        video_id = int(video_id)
    if isinstance(video_id, int):
        name = video_name_pattern.format(video_id)
    else:
        name = str(video_id)
    return os.path.join(video_root, f"{name}{video_ext}")
