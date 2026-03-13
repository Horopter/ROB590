# Dataset Label Descriptions

*Add these to the Corso Group SurgVLM Dataset Descriptions Google Doc.*

---

## Cholec80-CVS

**Source:** Ríos et al. (2023). Cholec80-CVS: An open dataset with an evaluation of Strasberg's critical view of safety for AI. *Scientific Data* 10, 194. https://doi.org/10.1038/s41597-023-02073-7

**Dataset:** Annotations for Strasberg's Critical View of Safety (CVS) criteria on all 80 videos from the Cholec80 laparoscopic cholecystectomy dataset.

**Columns:** Video, Critical View, Initial/Final Minute/Second, Two Structures, Cystic Plate, Hepatocystic Triangle, Total. Critical view achieved when Total ≥ 5.

---

## CholecSeg8k

**Source:** Hong et al. (2020). CholecSeg8k: A Semantic Segmentation Dataset for Laparoscopic Cholecystectomy Based on Cholec80. https://arxiv.org/abs/2012.12453

**Dataset:** 8,080 RGB frames from 17 Cholec80 videos, pixel-level semantic segmentation. 854×480 resolution.

### Class IDs (13 classes)

| ID | Class Name |
|----|------------|
| 0 | Black Background |
| 1 | Abdominal Wall |
| 2 | Liver |
| 3 | Gastrointestinal Tract |
| 4 | Fat |
| 5 | Grasper |
| 6 | Connective Tissue |
| 7 | Blood |
| 8 | Cystic Duct |
| 9 | L-hook Electrocautery |
| 10 | Gallbladder |
| 11 | Hepatic Vein |
| 12 | Liver Ligament |

---

## CholecTrack20

**Source:** Nwoye et al. (2025). CholecTrack20: A Multi-Perspective Tracking Dataset for Surgical Tools. CVPR 2025. https://arxiv.org/abs/2312.07352

**Dataset:** 20 videos, 35K+ frames at 1 fps, 65K+ tool instances. Annotations at Synapse.org.

### Tool Categories (7 classes)

| ID | Category |
|----|----------|
| 0 | Grasper |
| 1 | Bipolar |
| 2 | Hook |
| 3 | Scissors |
| 4 | Clipper |
| 5 | Irrigator |
| 6 | Specimen Bag |

### Annotation Attributes

- **tool_bbox**: [x, y, w, h] bounding box
- **instrument**: Tool category ID
- **operator**: MSLH, MSRH, ASRH, NULL
- **phase**: Surgical phase
- **intraoperative_track_id**, **intracorporeal_track_id**, **visibility_track_id**: Multi-perspective tracking IDs

---

## CholecT50

**Source:** Nwoye et al. (2022). Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos. Medical Image Analysis. https://doi.org/10.1016/j.media.2022.102433

**Dataset:** 50 videos, ~100K frames at 1 fps. Action triplets &lt;instrument, verb, target&gt;, phase labels.

### Annotation Format

- **Triplet**: &lt;instrument, verb, target&gt; (100 categories)
- **Phase**: Surgical phase per frame
- **Bounding box** (5 videos): BX, BY, BW, BH (normalized) for instrument tips

### Categories

- **Instruments**: 7 (grasper, bipolar, hook, scissors, clipper, irrigator, specimen bag)
- **Verbs**: 10 (e.g., grasp, retract, cauterize)
- **Targets**: 15 (e.g., gallbladder, cystic duct, liver)
