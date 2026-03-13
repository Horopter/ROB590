# Dataloader Verification Report

**Date:** March 13, 2025

## Summary

| Dataset | Downloaded | Verified | Notes |
|---------|------------|----------|-------|
| Cholec80-CVS | ✅ (xlsx in project) | ✅ | 568 segments after filtering invalid |
| CholecSeg8k | ✅ (HuggingFace) | ✅ | 8,080 frames, 854×480 |
| CholecTrack20 | ⚠️ Fixtures only | ✅ | Requires Synapse.org form for full data |
| CholecT50 | ⚠️ Fixtures only | ✅ | Requires CAMMA form for full data |

## Download Instructions

### Cholec80-CVS
- **Annotations:** `cholec80-CVS.xlsx` (in project) or from [Figshare](https://doi.org/10.6084/m9.figshare.c.5880458.v1)
- **Videos:** Request from [CAMMA](http://camma.u-strasbg.fr/datasets) (Cholec80)

### CholecSeg8k
```bash
# Option 1: HuggingFace (3.1 GB)
curl -L -o CholecSeg8k.zip "https://huggingface.co/datasets/minwoosun/CholecSeg8k/resolve/main/data/CholecSeg8k.zip?download=true"
unzip CholecSeg8k.zip -d data_downloads/

# Option 2: Kaggle (requires kaggle CLI + API key)
kaggle datasets download -d newslab/cholecseg8k
unzip cholecseg8k.zip -d data_downloads/
```

### CholecTrack20
1. Complete [request form](https://docs.google.com/forms/d/e/1FAIpQLSdewhAi0vGmZj5DLOMWdLf85BhUtTedS28YzvHS58ViwuEX5w/viewform)
2. Download from [Synapse.org](https://www.synapse.org/Synapse:syn53182642/wiki/)

### CholecT50
1. Visit [CholecT50 downloads](https://github.com/CAMMA-public/cholect50/blob/master/docs/README-Downloads.md)
2. Complete CAMMA request form

## Verification Command

```bash
python scripts/verify_dataloaders.py
```

## CholecSeg8k Mask Values

The paper defines 13 classes (IDs 0–12). Observed mask values may include 255 (ignore/unlabeled). For training, consider mapping or clamping values to 0–12 as needed.
