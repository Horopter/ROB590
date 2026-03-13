# ROB590 – Surgical Video Dataloaders

Production-ready data loaders for surgical video datasets, built with OOP inheritance.

**Datasets:** AVOS, Cholec80-CVS, CholecSeg8k, CholecTrack20, CholecT50

**Architecture:** Base classes in `dataloaders/base.py`, shared I/O in `dataloaders/io.py`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Cholec80-CVS Dataloader

Loads Strasberg's Critical View of Safety (CVS) annotations from the Cholec80-CVS dataset. Compatible with the [CHOLEC80-CVS-PUBLIC](https://github.com/ManuelRios18/CHOLEC80-CVS-PUBLIC) pipeline.

**Dataset:** https://doi.org/10.1038/s41597-023-02073-7  
**Annotations:** XLSX (surgeons_annotations.xlsx) or CSV from Figshare

### Option 1: Raw XLSX – Temporal segments (video clips)

Download annotations from [Figshare](https://doi.org/10.6084/m9.figshare.c.5880458.v1) or use `surgeons_annotations.xlsx`. Then:

```python
from torch.utils.data import DataLoader
from dataloaders.cholec80_cvs import (
    Cholec80CVSTemporalDataset,
    cholec80_cvs_temporal_collate_fn,
)

dataset = Cholec80CVSTemporalDataset(
    annotations_path="cholec80-CVS.xlsx",  # from Figshare
    video_root="./videos",  # Cholec80: video01.mp4, video02.mp4, ...
    num_frames=16,
    task="classification",
    cvs_threshold=5,
)
loader = DataLoader(dataset, batch_size=4, collate_fn=cholec80_cvs_temporal_collate_fn)
```

### Option 2: Processed frames (CHOLEC80-CVS-PUBLIC pipeline)

After running the [CHOLEC80-CVS-PUBLIC](https://github.com/ManuelRios18/CHOLEC80-CVS-PUBLIC) data preprocessing (`get_valid_frames.py`, `video_2_frames.py`, `annotations_2_labels.py`, `get_training_sets.py`):

```python
from dataloaders.cholec80_cvs import (
    Cholec80CVSFrameDataset,
    cholec80_cvs_frame_collate_fn,
)

dataset = Cholec80CVSFrameDataset(
    set_name="train",  # or "val", "test"
    frames_root="/path/to/extracted/frames",
    data_dir="data",
)
loader = DataLoader(dataset, batch_size=32, collate_fn=cholec80_cvs_frame_collate_fn)
```

### Video Naming

Videos are expected as `video01.mp4`, `video02.mp4`, ..., `video80.mp4`. Customize with:

```python
dataset = Cholec80CVSTemporalDataset(
    annotations_path="...",
    video_root="./videos",
    video_name_pattern="cholec80_{:02d}",  # cholec80_01.mp4, etc.
)
```

## CholecSeg8k Dataloader

Semantic segmentation for laparoscopic cholecystectomy. 8,080 frames, 13 classes.

**Dataset:** https://www.kaggle.com/datasets/newslab/cholecseg8k

```python
from dataloaders.cholecseg8k import CholecSeg8kDataset, cholecseg8k_collate_fn

dataset = CholecSeg8kDataset(root="./CholecSeg8k")
loader = DataLoader(dataset, batch_size=8, collate_fn=cholecseg8k_collate_fn)
```

## CholecTrack20 Dataloader

Multi-perspective surgical tool tracking (detection). 20 videos, 35K+ frames, 7 tool categories.

**Dataset:** https://github.com/camma-public/cholectrack20

```python
from dataloaders.cholectrack20 import CholecTrack20DetectionDataset, cholectrack20_detection_collate_fn

dataset = CholecTrack20DetectionDataset(root="./CholecTrack20", split="train")
loader = DataLoader(dataset, batch_size=4, collate_fn=cholectrack20_detection_collate_fn)
```

## CholecT50 Dataloader

Surgical action triplet recognition: &lt;instrument, verb, target&gt;. 50 videos, ~100 triplet categories.

**Dataset:** https://github.com/CAMMA-public/cholect50

```python
from dataloaders.cholect50 import CholecT50TripletDataset, cholect50_collate_fn

dataset = CholecT50TripletDataset(root="./CholecT50")
loader = DataLoader(dataset, batch_size=8, collate_fn=cholect50_collate_fn)
```

## AVOS Dataloader

From Andre's implementation. Supports object detection (bounding boxes) and temporal action recognition.

### Detection

```python
from dataloaders.avos import AVOSBoundingBoxDataset, detection_collate_fn

dataset = AVOSBoundingBoxDataset("open_surgery_bboxes_Jan16.csv", "./images")
loader = DataLoader(dataset, batch_size=4, collate_fn=detection_collate_fn)
```

### Temporal Action Recognition

```python
from dataloaders.avos import AVOSTemporalActionDataset, temporal_collate_fn

dataset = AVOSTemporalActionDataset("open_surgery_temporal_annotations_Jan16.csv", "./videos")
loader = DataLoader(dataset, batch_size=2, collate_fn=temporal_collate_fn)
```

## Tests

```bash
pytest tests/ -v
```

## Verification

After downloading datasets to `data_downloads/`, run:

```bash
python scripts/verify_dataloaders.py
```

See [docs/VERIFICATION_REPORT.md](docs/VERIFICATION_REPORT.md) for download instructions and verification results.

## Label Descriptions

See `docs/DATASET_LABEL_DESCRIPTIONS.md` for label descriptions (Cholec80-CVS, CholecSeg8k, CholecTrack20, CholecT50) to add to the shared Google Doc.
