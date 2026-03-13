#!/usr/bin/env python3
"""
Verify dataloaders against downloaded datasets.

Run after downloading datasets to data_downloads/ or configuring paths below.

Datasets:
- Cholec80-CVS: Use cholec80-CVS.xlsx (in project root). Videos from CAMMA.
- CholecSeg8k: Download from https://huggingface.co/datasets/minwoosun/CholecSeg8k
  or: kaggle datasets download -d newslab/cholecseg8k
- CholecTrack20: Request access at Synapse.org (form required)
- CholecT50: Request via CAMMA cholect50 GitHub
"""

import os
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "data_downloads"


def verify_cholec80_cvs():
    """Verify Cholec80-CVS dataloader."""
    xlsx = PROJECT_ROOT / "cholec80-CVS.xlsx"
    if not xlsx.exists():
        print("SKIP Cholec80-CVS: cholec80-CVS.xlsx not found")
        return False

    from unittest.mock import patch
    import torch
    from dataloaders.cholec80_cvs import Cholec80CVSTemporalDataset

    ds = Cholec80CVSTemporalDataset(
        annotations_path=str(xlsx),
        video_root="/tmp",
        skip_invalid_segments=True,
    )
    assert len(ds) > 0, "Empty dataset"
    with patch("dataloaders.base.read_video_clip", return_value=torch.rand(16, 3, 224, 224)):
        clip, label, meta = ds[0]
    assert clip.shape == (16, 3, 224, 224)
    assert "video_id" in meta and "total" in meta
    print(f"  Cholec80-CVS: OK (len={len(ds)})")
    return True


def verify_cholecseg8k():
    """Verify CholecSeg8k dataloader."""
    root = DATA_ROOT / "CholecSeg8k"
    if not root.exists():
        root = DATA_ROOT / "CholecSeg8k"  # after unzip
    if not root.exists():
        alt = PROJECT_ROOT / "CholecSeg8k"
        root = alt if alt.exists() else None
    if root is None or not root.exists():
        print("SKIP CholecSeg8k: Extract CholecSeg8k.zip to data_downloads/")
        return False

    from dataloaders.cholecseg8k import CholecSeg8kDataset, cholecseg8k_collate_fn

    ds = CholecSeg8kDataset(root=str(root))
    assert len(ds) == 8080, f"Expected 8080 samples, got {len(ds)}"
    img, mask = ds[0]  # Single sample load
    assert img.shape == (3, 480, 854)
    assert mask.shape == (480, 854)
    print(f"  CholecSeg8k: OK (len={len(ds)}, img={img.shape}, mask={mask.shape})")
    return True


def verify_cholectrack20():
    """Verify CholecTrack20 dataloader."""
    root = DATA_ROOT / "CholecTrack20"
    if not root.exists():
        root = PROJECT_ROOT / "tests" / "fixtures" / "cholectrack20"
    if not root.exists():
        print("SKIP CholecTrack20: Download from Synapse.org (form required)")
        return False

    from dataloaders.cholectrack20 import CholecTrack20DetectionDataset

    for split in ["train", "val", "test"]:
        ds = CholecTrack20DetectionDataset(root=str(root), split=split)
        if len(ds) > 0:
            img, target = ds[0]
            assert "boxes" in target
            print(f"  CholecTrack20 ({split}): OK (len={len(ds)})")
            return True
    print("SKIP CholecTrack20: No data in splits")
    return False


def verify_cholect50():
    """Verify CholecT50 dataloader."""
    root = DATA_ROOT / "CholecT50"
    if not root.exists():
        root = PROJECT_ROOT / "CholecT50"
    if not root.exists():
        root = PROJECT_ROOT / "tests" / "fixtures" / "cholect50"
    if not root.exists():
        print("SKIP CholecT50: Download from CAMMA cholect50 (form required)")
        return False

    from dataloaders.cholect50 import CholecT50TripletDataset

    ds = CholecT50TripletDataset(root=str(root))
    assert len(ds) > 0
    img, triplets, phase = ds[0]
    assert isinstance(triplets, list)
    print(f"  CholecT50: OK (len={len(ds)})")
    return True


def main():
    print("Verifying dataloaders...")
    results = {}
    results["Cholec80-CVS"] = verify_cholec80_cvs()
    results["CholecSeg8k"] = verify_cholecseg8k()
    results["CholecTrack20"] = verify_cholectrack20()
    results["CholecT50"] = verify_cholect50()

    passed = sum(results.values())
    total = len(results)
    print(f"\nResult: {passed}/{total} dataloaders verified")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
