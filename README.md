# Image Matching & Verification System

A computer vision pipeline for verifying whether two photographs show the exact same physical stone piece or distinct slabs. Slabs cut from the same quarry block share texture and coloration ("family resemblance"), but have distinct micro-vein geometry ("fingerprints"). This system enforces rigid geometric verification to identify matching pieces while rejecting lookalikes.

---

## Pipeline Overview

1. **Preprocessing:** Aspect-ratio-preserving resize to bounded dimension ($1024\text{ px}$ max edge), grayscale conversion, and optional CLAHE contrast enhancement.
2. **Feature Extraction:** Scale- and rotation-invariant keypoint detection using SIFT and RootSIFT ($L_1$ normalization followed by square root).
3. **Feature Matching:** $L_2$ distance nearest-neighbor search with Lowe's ratio test ($0.75$) to discard ambiguous texture matches.
4. **Geometric Verification:** Planar homography estimation via robust RANSAC/MAGSAC, inlier filtering, and spatial dispersion checks.
5. **Calibrated Scoring:** Confidence score ($0\text{--}100\%$) based on inlier count, inlier ratio, and spatial coverage, mapped into three decision bands: `SAME`, `SIMILAR`, and `DIFFERENT`.

---

## Quickstart

### 1. Installation

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows (.venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

### 2. Compare a Single Pair

```bash
python match.py --image-a Intern_Assignment_Dataset/reference/img_036__B01A_carrara_reference.jpg --image-b Intern_Assignment_Dataset/queries/img_021__B01A_rotation_scale_same.jpg --out output/
```

Outputs metrics to console (keypoints, inliers, inlier ratio, confidence score, decision band) and saves an inlier-only visualization to `output/`.

### 3. Batch Evaluation

```bash
python evaluate.py --data Intern_Assignment_Dataset --pairs Intern_Assignment_Dataset/pairs.csv --out results.csv
```

Runs all pairs through the pipeline, caches extracted features, writes detailed metrics to `results.csv`, and prints a breakdown table.

---

## Repository Structure

```
├── docs/                        # Project documentation
│   ├── BRD.md                   # Business requirements & success metrics
│   ├── PRD.md                   # Product specification & functional requirements
│   ├── ARCHITECTURE.md          # Technical design, algorithms & contracts
│   └── AGILE_ROADMAP.md         # Sprint plan and development milestones
├── src/                         # Core pipeline modules
│   ├── config.py                # Pipeline parameters and decision thresholds
│   ├── preprocessor.py          # Image loading, resizing, and enhancement
│   ├── extractor.py             # SIFT/RootSIFT extraction and feature cache
│   ├── matcher.py               # Feature matching and RANSAC homography
│   ├── scoring.py               # Confidence calculation and band decider
│   ├── visualizer.py            # Inlier match rendering
│   └── image_matcher.py         # Unified match_pair() API
├── Intern_Assignment_Dataset/   # Benchmark dataset (5 references x 20 queries)
├── tests/                       # Automated test suites
├── match.py                     # Single-pair CLI tool
├── evaluate.py                  # Batch evaluation CLI tool
├── requirements.txt             # Pinned dependencies
└── README.md
```

---

## Running Tests

Run the verification test suite:

```bash
python tests/test_sprint1.py
```
