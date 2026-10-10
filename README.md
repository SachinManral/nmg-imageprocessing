# Precision Stone Slab Image Matching & Verification System

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8.svg)](https://opencv.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Accuracy](https://img.shields.io/badge/Benchmark%20Accuracy-100%25-success.svg)](#benchmark-performance--validation)
[![Lookalike FAR](https://img.shields.io/badge/Lookalike%20FAR-0.0%25-brightgreen.svg)](#benchmark-performance--validation)
[![Air-Gapped](https://img.shields.io/badge/Architecture-100%25%20Offline-orange.svg)](#architectural-overview)

A robust, offline computer vision pipeline engineered to verify whether two photographs depict the **exact same physical stone slab** across complex photometric and geometric transformations (rotations, scale changes, oblique viewpoints, non-uniform glare, and up to 55% partial occlusion).

The system solves the high-stakes **"Fingerprint vs. Family Resemblance"** challenge in stone logistics: distinguishing identical slabs from sibling slabs quarried inches apart from the same block (**lookalikes**), enforcing zero false acceptances while operating with low latency.

---

## Table of Contents

- [The Core Challenge: Fingerprint vs. Family Resemblance](#the-core-challenge-fingerprint-vs-family-resemblance)
- [Architectural Overview](#architectural-overview)
- [Pipeline Engineering & Mathematical Foundations](#pipeline-engineering--mathematical-foundations)
  - [1. Preprocessing & CLAHE Ablation](#1-preprocessing--clahe-ablation)
  - [2. Feature Extraction: Low-Threshold SIFT](#2-feature-extraction-low-threshold-sift)
  - [3. Hellinger Kernel: RootSIFT Formulation](#3-hellinger-kernel-rootsift-formulation)
  - [4. Ambiguity Filtering: Lowe's Ratio Test](#4-ambiguity-filtering-lowes-ratio-test)
  - [5. Planar Homography & USAC_MAGSAC](#5-planar-homography--usac_magsac)
  - [6. Spatial Coverage & Degeneracy Guard](#6-spatial-coverage--degeneracy-guard)
  - [7. Multi-Factor Scoring & Calibrated Triad Bands](#7-multi-factor-scoring--calibrated-triad-bands)
- [Decision Triad & Classification Matrix](#decision-triad--classification-matrix)
- [Benchmark Performance & Validation](#benchmark-performance--validation)
- [Repository Structure](#repository-structure)
- [Quickstart Guide](#quickstart-guide)
  - [1. Installation](#1-installation)
  - [2. Single-Pair Verification CLI (`match.py`)](#2-single-pair-verification-cli-matchpy)
  - [3. Batch Benchmark CLI (`evaluate.py`)](#3-batch-benchmark-cli-evaluatepy)
  - [4. Parameter Ablation Study (`tune.py`)](#4-parameter-ablation-study-tunepy)
  - [5. Interactive Web Dashboard & REST Server](#5-interactive-web-dashboard--rest-server)
- [REST API Reference](#rest-api-reference)
- [Automated Test Suite](#automated-test-suite)
- [Engineering Trade-offs & Failure Mode Analysis](#engineering-trade-offs--failure-mode-analysis)

---

## The Core Challenge: Fingerprint vs. Family Resemblance

Natural stones (e.g., Carrara marble, Statuario, Botticino, and Nero granite) present a unique computer vision hurdle:

* **Family Resemblance (Sibling Lookalikes):** Slabs cut consecutively from the same quarried block share the exact same background mineral color, crystalline texture density, and general vein character. Standard deep neural network embeddings, histogram comparators, and bag-of-words models routinely fail here by declaring sibling pieces to be identical.
* **Unique Fingerprint (Physical Slabs):** The micro-vein branching paths and natural fissures form a unique 2D planar fingerprint.
* **The Goal:** Enforce strict planar homography consensus so that lookalikes (which lack identical vein geometry) are rejected with a **0.0% False Acceptance Rate (FAR)**, while genuine pairs are accepted across rotations, tilts, shadows, and partial occlusions.

---

## Architectural Overview

The following diagram illustrates the complete end-to-end verification and evaluation architecture:

```mermaid
flowchart TD
    subgraph INGESTION["1. Image Ingestion & Preprocessing"]
        ImgA["Image A (Reference / Catalog)"] --> PrepA["Aspect-Preserving Resize (Max 1024px)<br/>Grayscale Conversion"]
        ImgB["Image B (Query / Inspection)"] --> PrepB["Aspect-Preserving Resize (Max 1024px)<br/>Grayscale Conversion"]
    end

    subgraph FEATURES["2. Feature Detection & Representation"]
        PrepA --> SIFT_A["Low-Threshold SIFT<br/>(contrastThreshold = 0.01)"]
        PrepB --> SIFT_B["Low-Threshold SIFT<br/>(contrastThreshold = 0.01)"]
        SIFT_A --> RS_A["RootSIFT Mapping<br/>(L1 Norm + Element-wise Sqrt)"]
        SIFT_B --> RS_B["RootSIFT Mapping<br/>(L1 Norm + Element-wise Sqrt)"]
        CACHE[("FeatureCache<br/>(Path + mtime Keyed)")] -.-> RS_A
        CACHE -.-> RS_B
    end

    subgraph MATCHING["3. Correspondence & Geometric Verification"]
        RS_A & RS_B --> BF["Brute-Force L2 Matcher<br/>k-Nearest Neighbors (k=2)"]
        BF --> LOWE["Lowe's Ratio Test (τ = 0.75)<br/>Ambiguity & Granular Noise Filter"]
        LOWE --> MAGSAC["cv2.findHomography<br/>USAC_MAGSAC (Threshold = 4.0px)"]
        MAGSAC --> SPATIAL{"Spatial Dispersion Check<br/>Convex Hull Coverage ≥ 2%"}
    end

    subgraph SCORING["4. Multi-Factor Scoring & Decision Bands"]
        SPATIAL -- Passed --> ENGINE["Calibrated Scoring Formula<br/>Count Saturation (65%) + Ratio (35%)<br/>× Spatial Dampener"]
        SPATIAL -- Failed --> ZERO["Zero Confidence Score (0.0%)"]
        ENGINE --> TRIAD{"Decision Thresholds"}
        ZERO --> BAND_DIFF["DIFFERENT<br/>(Rejected / Lookalike)"]
        TRIAD -- "Conf ≥ 60%, Inliers ≥ 35, Ratio ≥ 0.30" --> BAND_SAME["SAME<br/>(Verified Physical Slab)"]
        TRIAD -- "Conf ≥ 25% or Inliers ≥ 10" --> BAND_SIM["SIMILAR<br/>(Partial / Inconclusive)"]
        TRIAD -- "Inliers < 10 or Low Conf" --> BAND_DIFF
    end

    subgraph OUTPUTS["5. Deliverables & Interfaces"]
        BAND_SAME & BAND_SIM & BAND_DIFF --> CLI["Single-Pair CLI (match.py)"]
        BAND_SAME & BAND_SIM & BAND_DIFF --> BATCH["Batch Evaluation (evaluate.py)"]
        BAND_SAME & BAND_SIM & BAND_DIFF --> API["FastAPI REST Server (server.py)"]
        API --> UI["Web App (1-to-1 & 1-to-N Search)"]
        BAND_SAME & BAND_SIM & BAND_DIFF --> VIZ["Inlier Overlay JPG Export"]
    end

    classDef success fill:#d1e7dd,stroke:#0f5132,stroke-width:1.5px,color:#0f5132;
    classDef warning fill:#fff3cd,stroke:#664d03,stroke-width:1.5px,color:#664d03;
    classDef danger fill:#f8d7da,stroke:#842029,stroke-width:1.5px,color:#842029;
    classDef highlight fill:#e7f1ff,stroke:#0d6efd,stroke-width:1.5px,color:#0d6efd;

    class BAND_SAME success;
    class BAND_SIM warning;
    class BAND_DIFF danger;
    class ENGINE,MAGSAC,RS_A,RS_B highlight;
```

---

## Pipeline Engineering & Mathematical Foundations

### 1. Preprocessing & CLAHE Ablation
- **Aspect-Ratio-Preserving Resize:** Longest edge constrained to $\le 1024\text{ px}$. This retains ultra-fine vein contours while capping processing latency to $< 250\text{ ms}$.
- **Grayscale Conversion:** Isolates luminance gradients from environmental color temperature shifts (e.g., warm indoor warehouse lighting vs. natural daylight).
- **CLAHE Ablation Decision:** Contrast-Limited Adaptive Histogram Equalization was empirically evaluated. On speckled granites, CLAHE boosted micro-noise by $300\%$, increasing nearest-neighbor matching time by $180\%$ without improving lookalike discrimination. Consequently, **CLAHE is disabled by default** (`use_clahe=False`).

### 2. Feature Extraction: Low-Threshold SIFT
- Standard OpenCV SIFT relies on `contrastThreshold = 0.04`, which filters out subtle veins on polished white Carrara or Statuario marble ($< 80$ keypoints detected).
- We calibrated `contrastThreshold = 0.01` with `edgeThreshold = 10.0`, yielding $3,500\text{--}4,000$ keypoints on smooth marble while capturing faint vein bifurcations.
- `n_features = 6000` prevents combinatorial runtime explosion on heavily speckled granite slabs.

### 3. Hellinger Kernel: RootSIFT Formulation
Standard SIFT uses Euclidean distance ($L_2$) on 128-dimensional gradient orientation histograms. $L_2$ implicitly assumes a Gaussian error model that gives disproportionate weight to large bins.

The **Hellinger distance** for histograms $x, y \ge 0$ is defined as:
$$d_H(x, y) = \sqrt{1 - \sum_{i=1}^{128} \sqrt{x_i y_i}}$$

RootSIFT algebraically maps the Hellinger distance into standard Euclidean space:
1. **$L_1$ Normalization:**
   $$\tilde{x}_i = \frac{x_i}{\sum_{k=1}^{128} |x_k| + \epsilon}$$
2. **Element-wise Square Root:**
   $$z_i = \sqrt{\tilde{x}_i}$$
3. **Inherent Unit $L_2$ Norm:**
   $$\|z\|_2^2 = \sum_{i=1}^{128} (\sqrt{\tilde{x}_i})^2 = \sum_{i=1}^{128} \tilde{x}_i = 1$$
   Because $\|z\|_2 = 1$ is naturally satisfied, secondary $L_2$ normalization is mathematically redundant and omitted.

Empirical testing across 100 pairs demonstrated that RootSIFT expands the safety margin between the weakest genuine match and strongest lookalike by **$+13$ inliers ($+4.9\%$)**.

### 4. Ambiguity Filtering: Lowe's Ratio Test
For each descriptor $d_A$, let $n_{B,1}$ and $n_{B,2}$ be the first and second nearest neighbors with distances $r_1 \le r_2$:
$$\text{Keep match} \iff \frac{r_1}{r_2} \le \tau_{\text{ratio}} \quad (\tau_{\text{ratio}} = 0.75)$$
This prunes repetitive crystalline patterns where ambiguous neighboring crystals produce nearly equidistant candidate matches.

### 5. Planar Homography & USAC_MAGSAC
Because stone slabs are rigid planar surfaces, any two views of the same slab obey a projective planar homography $H \in \mathbb{R}^{3 \times 3}$:
$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

We employ `cv2.findHomography` with `cv2.USAC_MAGSAC` (reprojection error $\tau_{\text{reproj}} = 4.0\text{ px}$, confidence $= 0.999$, $\text{maxIters} = 3000$). MAGSAC++ provides robust marginalization over noise scale, outperforming standard RANSAC under oblique viewing angles.

### 6. Spatial Coverage & Degeneracy Guard
A high inlier count clustered within a tight 20-pixel area can yield a mathematically valid homography that is physically degenerate. We compute the normalized convex hull area of inlier keypoints on both images:
$$\text{Coverage} = \min\left( \frac{\text{Area}(\text{ConvexHull}_A)}{W_A \cdot H_A}, \; \frac{\text{Area}(\text{ConvexHull}_B)}{W_B \cdot H_B} \right)$$
- If $\text{Coverage} < 0.02$ ($2\%$ of image area) or $N_{\text{inliers}} < 8$, geometric consensus is declared **invalid**.
- If $\text{Coverage} < 0.05$ ($5\%$), a linear dampener penalizes the confidence score: $\text{Penalty} = \max(0.3, \text{Coverage} / 0.05)$.

### 7. Multi-Factor Scoring & Calibrated Triad Bands
When geometric verification is valid, the confidence score $S \in [0.0, 100.0]\%$ combines inlier volume and ratio consistency:
$$f_{\text{count}} = \min\left(1.0, \frac{N_{\text{inliers}}}{50.0}\right), \quad f_{\text{ratio}} = \frac{N_{\text{inliers}}}{M_{\text{good}}}$$
$$S_{\text{raw}} = (0.65 \cdot f_{\text{count}} + 0.35 \cdot f_{\text{ratio}}) \times 100.0$$
$$S = S_{\text{raw}} \times \text{CoveragePenalty}$$

---

## Decision Triad & Classification Matrix

| Decision Band | Mathematical Criteria | Physical Interpretation | Operational Action |
|:---|:---|:---|:---|
| <span style="color:#0f5132;font-weight:bold;">SAME</span> | $\text{Conf} \ge 60.0\%$ **AND** $N_{\text{inliers}} \ge 35$ **AND** $R_{\text{inlier}} \ge 0.30$ | Verified exact physical slab. Planar vein alignment mathematically confirmed. | Automated dispatch / approval; zero human review needed. |
| <span style="color:#664d03;font-weight:bold;">SIMILAR</span> | $\text{Conf} \ge 25.0\%$ **OR** $N_{\text{inliers}} \ge 10$ | Inconclusive evidence. Likely extreme occlusion ($> 50\%$) or severe viewpoint angle. | Flag for secondary manual inspection; do not reject outright. |
| <span style="color:#842029;font-weight:bold;">DIFFERENT</span> | $N_{\text{inliers}} < 10$ **OR** Geometry Fails ($\text{Conf} = 0.0\%$) | Distinct stone or sibling lookalike slab. No consistent planar homography. | Automatic rejection; prevent lookalike inventory mismatch. |

---

## Benchmark Performance & Validation

Evaluated across the benchmark suite (5 Reference Slabs $\times$ 20 Query Slabs = 100 Pairwise Comparisons):

### Summary Accuracy

| Metric | Target Specification | Measured Result | Margin / Status |
|:---|:---:|:---:|:---:|
| **Same Pairs Classified `SAME`** | $\ge 90.0\%$ | **10 / 10 (100.0%)** | $+10.0\%$ (Perfect Recall) |
| **Different Pairs Correctly Rejected** | $\ge 95.0\%$ | **90 / 90 (100.0%)** | $+5.0\%$ (Perfect Rejection) |
| **Lookalike False Acceptances** | **0** | **0 / 10 (0 FAR)** | **Zero False Acceptances** |
| **Weakest Genuine Match ($N_{\text{inliers}}$)** | $> 40$ inliers | **284 inliers** | $7.1\times$ above minimum threshold |
| **Strongest Lookalike Sibling ($N_{\text{inliers}}$)** | $< 15$ inliers | **6 inliers** | Safety Separation Gap = **$47\times$** |
| **Average Latency per Pair** | $< 400\text{ ms}$ | **225.8 ms** | $43.5\%$ under latency budget |

### Breakdown by Perturbation Type

| Category | Total Pairs | Ground Truth | Classified `SAME` | Classified `SIMILAR` | Classified `DIFFERENT` | Pass Rate |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `rotation_scale` | 3 | Same | 3 | 0 | 0 | **100.0%** |
| `viewpoint` | 3 | Same | 3 | 0 | 0 | **100.0%** |
| `lighting` | 2 | Same | 2 | 0 | 0 | **100.0%** |
| `occlusion` | 2 | Same | 2 | 0 | 0 | **100.0%** |
| `lookalike` | 10 | Different | 0 | 0 | 10 | **100.0%** |
| `different` | 80 | Different | 0 | 0 | 80 | **100.0%** |

---

## Repository Structure

```
nmg/
├── src/                               # Core pipeline modules
│   ├── config.py                      # Immutable pipeline configurations & thresholds
│   ├── preprocessor.py                # Aspect-ratio resizing, grayscale conversion, CLAHE
│   ├── extractor.py                   # SIFT, RootSIFT transformation & FeatureCache
│   ├── matcher.py                     # BFMatcher, Lowe ratio test, USAC_MAGSAC homography
│   ├── scoring.py                     # Confidence score computation & 3-band classification
│   ├── visualizer.py                  # Inlier match rendering & base64 export
│   └── image_matcher.py               # Unified ImageMatcher class & match_pair() API
├── docs/                              # Project design & engineering specifications
│   ├── BRD.md                         # Business Requirements Document & Success Metrics
│   ├── PRD.md                         # Product Requirements Document & Functional Specs
│   ├── ARCHITECTURE.md                # System Architecture, Math & Algorithmic Contracts
│   └── AGILE_ROADMAP.md               # Sprints 1-4 execution roadmap & user stories
├── tests/                             # Automated regression & integration test suite
│   ├── test_sprint1.py                # Preprocessing, RootSIFT & caching unit tests
│   ├── test_sprint2.py                # Geometric verification & homography tests
│   ├── test_sprint3.py                # Scoring engine & end-to-end matching tests
│   ├── test_sprint4.py                # Benchmark integrity & live zero-hardcode tests
│   └── test_server.py                 # FastAPI endpoints & REST integration tests
├── frontend/                          # Clean, responsive web dashboard
│   ├── index.html                     # Tabbed UI (1-to-1 Verify, 1-to-N Search, Benchmark)
│   ├── style.css                      # Modern dark/light design system
│   └── app.js                         # REST API client, drag-and-drop, camera handling
├── Intern_Assignment_Dataset/         # Benchmark dataset
│   ├── reference/                     # 5 registered reference slabs
│   ├── queries/                       # 20 query variations (rotations, glare, lookalikes)
│   └── pairs.csv                      # Official 100-pair evaluation split
├── match.py                           # Single-pair CLI verification tool
├── evaluate.py                        # Batch evaluation CLI tool (writes results.csv)
├── tune.py                            # Parameter ablation & tuning script
├── server.py                          # FastAPI production server with CORS
├── REPORT.md                          # Detailed engineering report & failure analysis
├── requirements.txt                   # Pinned production dependencies
└── README.md                          # Main documentation
```

---

## Quickstart Guide

### 1. Installation

Create and activate a virtual environment, then install dependencies:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Single-Pair Verification CLI (`match.py`)

Compare any two stone images and optionally render an inlier visualization:

```bash
python match.py \
  --image-a Intern_Assignment_Dataset/reference/img_036__B01A_carrara_reference.jpg \
  --image-b Intern_Assignment_Dataset/queries/img_021__B01A_rotation_scale_same.jpg \
  --out output/
```

**Console Output:**
```text
Keypoints A: 3742
Keypoints B: 3918
Total matches: 3742
Good matches: 1542
RANSAC inliers: 1289
Inlier ratio: 0.8359
Confidence: 94.3%
Band: SAME
Processing time: 248.5 ms
Saved inlier visualization: output/img_036__B01A_carrara_reference__img_021__B01A_rotation_scale_same.jpg
```

### 3. Batch Benchmark CLI (`evaluate.py`)

Evaluate the full 100-pair benchmark dataset (or any custom CSV pair list):

```bash
python evaluate.py \
  --data Intern_Assignment_Dataset \
  --pairs Intern_Assignment_Dataset/pairs.csv \
  --out results.csv
```

Outputs metrics to console and writes full per-pair diagnostics into `results.csv`.

> [!NOTE]
> If evaluated on an unlabelled hidden test set where the `label` column is `unknown`, `evaluate.py` gracefully writes `results.csv` without failing.

### 4. Parameter Ablation Study (`tune.py`)

Run the SIFT vs. RootSIFT empirical comparison across all benchmark pairs:

```bash
python tune.py --pairs Intern_Assignment_Dataset/pairs.csv --data Intern_Assignment_Dataset
```

### 5. Interactive Web Dashboard & REST Server

Start the FastAPI application:

```bash
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

Then open `frontend/index.html` in your web browser (or serve via any static HTTP server):
- **Tab 1: Verify Slab:** Upload two images, or upload just one to automatically match it against all reference slabs in inventory. Supports phone camera capture.
- **Tab 2: Catalog Search (1-to-N):** Upload an unknown slab photo and rank all registered references by confidence score.
- **Tab 3: Benchmark Audit:** Inspect live benchmark accuracy statistics and browse evaluated pairs with filtering.

---

## REST API Reference

The server exposes standard JSON endpoints with Swagger UI at `http://127.0.0.1:8000/docs`:

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/api/health` | Service health status check. |
| `GET` | `/api/presets` | List registered reference slabs and query samples. |
| `POST` | `/api/match` | Compare Image A and B (file upload or preset path). Supports auto-matching against inventory if only one image is supplied. |
| `POST` | `/api/gallery-search` | 1-to-N catalog search: ranks all inventory references against a query image. |
| `GET` | `/api/benchmark-results` | Reads `results.csv` and returns aggregate accuracy statistics. |

---

## Automated Test Suite

Run the full pytest test suite covering all sprints and server endpoints:

```bash
python -m pytest tests/ -v
```

**Test Coverage Summary:**
- **`tests/test_sprint1.py`:** Aspect-ratio resizing, grayscale conversion, RootSIFT unit $L_2$ norm, and `FeatureCache` validation.
- **`tests/test_sprint2.py`:** Lowe's ratio test, MAGSAC homography estimation, lookalike rejection, and spatial coverage checks.
- **`tests/test_sprint3.py`:** Calibrated scoring formula and `SAME`/`SIMILAR`/`DIFFERENT` band decider.
- **`tests/test_sprint4.py`:** `results.csv` data integrity, documentation completeness, and live zero-hardcoding verification.
- **`tests/test_server.py`:** Healthcheck, presets, batch results, and matching endpoints.

---

## Engineering Trade-offs & Failure Mode Analysis

| Challenge / Failure Mode | Root Cause | Engineering Resolution |
|:---|:---|:---|
| **High-Key Polished Marble Keypoint Starvation** | Smooth white crystalline limestone has low intensity gradient variance, yielding $< 80$ keypoints under standard SIFT (`contrastThreshold = 0.04`). | Decreased `contrastThreshold` to `0.01`. Keypoint count rose to $3,500\text{--}4,000$, unlocking $> 1,200$ inliers on genuine matches under illumination shifts. |
| **Speckled Granite Combinatorial Explosion** | Dense mineral speckles in granite produced up to $17,800$ keypoints, blowing up $k$-NN matching time to $> 4.2\text{ s}$. | Capped keypoints via `n_features = 6000` and introduced `FeatureCache` keyed by file path and `mtime`, dropping latency to $\sim 225\text{ ms}$. |
| **Accidental Clustered Inlier Homographies** | Inliers localized within a tiny surface imperfection could produce a low-error homography despite the rest of the slab being different. | Added Convex Hull spatial dispersion validation: required inlier footprint to span $\ge 2\%$ of image area and applied a dampener for coverage $< 5\%$. |
| **Sibling Lookalikes (Quarry Block Neighbors)** | Sibling slabs have near-identical color and general grain density. | Enforced Lowe's ratio test ($\tau = 0.75$) with MAGSAC planar homography. Sibling lookalikes achieved at most 6 inliers, leaving a **$47\times$ safety gap** below genuine pairs (min 284 inliers). |

---

## License

This project is licensed under the Apache 2.0 License.
