# Precision Stone Slab Image Matching & Verification System

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8.svg)](https://opencv.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Benchmark Accuracy](https://img.shields.io/badge/Benchmark%20Accuracy-100%25-success.svg)](#benchmark-performance--validation)
[![Lookalike FAR](https://img.shields.io/badge/Lookalike%20FAR-0.0%25-brightgreen.svg)](#benchmark-performance--validation)
[![Architecture](https://img.shields.io/badge/Architecture-100%25%20Offline-orange.svg)](#system-architecture--pipeline-flow)

A robust, offline computer vision system engineered to verify whether two photographs depict the **exact same physical stone slab** across complex photometric and geometric transformations (rotations, scale changes, oblique viewpoints, non-uniform glare, and up to 55% partial occlusion).

The system solves the high-stakes **"Fingerprint vs. Family Resemblance"** challenge in stone logistics: distinguishing identical slabs from sibling slabs quarried inches apart from the same block (**lookalikes**), enforcing zero false acceptances while operating with low latency.

---

## Table of Contents

- [The Core Challenge: Fingerprint vs. Family Resemblance](#the-core-challenge-fingerprint-vs-family-resemblance)
- [System Architecture & Pipeline Flow](#system-architecture--pipeline-flow)
- [Verification Decision Logic & Lookalike Rejection](#verification-decision-logic--lookalike-rejection)
- [Pipeline Engineering & Mathematical Foundations](#pipeline-engineering--mathematical-foundations)
  - [1. Image Ingestion & CLAHE Ablation](#1-image-ingestion--clahe-ablation)
  - [2. Feature Detection: Low-Threshold SIFT](#2-feature-detection-low-threshold-sift)
  - [3. Hellinger Kernel: RootSIFT Formulation](#3-hellinger-kernel-rootsift-formulation)
  - [4. Ambiguity Filtering: Lowe's Ratio Test](#4-ambiguity-filtering-lowes-ratio-test)
  - [5. Planar Homography: USAC_MAGSAC Consensus](#5-planar-homography-usac_magsac-consensus)
  - [6. Spatial Coverage & Degeneracy Guard](#6-spatial-coverage--degeneracy-guard)
  - [7. Multi-Factor Scoring & Calibrated Triad Bands](#7-multi-factor-scoring--calibrated-triad-bands)
- [Decision Triad & Classification Matrix](#decision-triad--classification-matrix)
- [Benchmark Dataset Structure & Pair Mapping](#benchmark-dataset-structure--pair-mapping)
- [Benchmark Performance & Validation](#benchmark-performance--validation)
- [Repository Structure](#repository-structure)
- [Quickstart Guide](#quickstart-guide)
  - [1. Installation & Environment Setup](#1-installation--environment-setup)
  - [2. Interactive Web Dashboard & REST Server](#2-interactive-web-dashboard--rest-server)
  - [3. Single-Pair Verification CLI (`match.py`)](#3-single-pair-verification-cli-matchpy)
  - [4. Batch Benchmark Evaluation CLI (`evaluate.py`)](#4-batch-benchmark-evaluation-cli-evaluatepy)
  - [5. Parameter Ablation Study (`tune.py`)](#5-parameter-ablation-study-tunepy)
- [REST API Reference](#rest-api-reference)
- [Automated Test Suite](#automated-test-suite)
- [Engineering Trade-offs & Failure Mode Analysis](#engineering-trade-offs--failure-mode-analysis)

---

## The Core Challenge: Fingerprint vs. Family Resemblance

Natural stone (e.g., Carrara marble, Statuario, Botticino, Nero granite, Grey granite) poses a unique computer vision challenge:

* **Family Resemblance (Sibling Lookalikes):** Slabs cut consecutively from the same quarried block share identical background mineral color, crystalline texture density, and general vein character. Standard deep neural network embeddings, color histograms, and appearance models fail by declaring sibling pieces to be identical.
* **Unique Fingerprint (Physical Slabs):** The micro-vein branching paths and natural fissures form a unique 2D planar fingerprint.
* **The Core Objective:** Enforce strict planar homography consensus so that lookalikes (which lack identical vein geometry) are rejected with a **0.0% False Acceptance Rate (FAR)**, while genuine pairs are verified across rotations, tilts, shadows, and partial occlusions.

---

## System Architecture & Pipeline Flow

The following diagram illustrates the complete end-to-end verification and evaluation architecture:

```mermaid
flowchart TD
    subgraph INGESTION["1. Ingestion & Preprocessing"]
        ImgA["Image A (Reference / Catalog)"] --> PrepA["Aspect-Preserving Resize (Max 1024px)<br/>Grayscale Luminance Conversion"]
        ImgB["Image B (Query / Inspection)"] --> PrepB["Aspect-Preserving Resize (Max 1024px)<br/>Grayscale Luminance Conversion"]
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

    subgraph SCORING["4. Multi-Factor Scoring & Decision Engine"]
        SPATIAL -- Passed --> ENGINE["Calibrated Confidence Formula<br/>Count Saturation (65%) + Ratio (35%)<br/>× Spatial Coverage Dampener"]
        SPATIAL -- Failed --> ZERO["Zero Confidence Score (0.0%)"]
        ENGINE --> TRIAD{"Decision Thresholds"}
        ZERO --> BAND_DIFF["DIFFERENT<br/>(Rejected / Lookalike Sibling)"]
        TRIAD -- "Conf ≥ 60%, Inliers ≥ 35, Ratio ≥ 0.30" --> BAND_SAME["SAME<br/>(Verified Physical Slab)"]
        TRIAD -- "Conf ≥ 25% or Inliers ≥ 10" --> BAND_SIM["SIMILAR<br/>(Partial / Inconclusive)"]
        TRIAD -- "Inliers < 10 or Low Conf" --> BAND_DIFF
    end

    subgraph PRESENTATION["5. Delivery & User Interfaces"]
        BAND_SAME & BAND_SIM & BAND_DIFF --> CLI["Single-Pair CLI (match.py)"]
        BAND_SAME & BAND_SIM & BAND_DIFF --> BATCH["Batch Evaluation (evaluate.py)"]
        BAND_SAME & BAND_SIM & BAND_DIFF --> API["FastAPI REST Server (server.py)"]
        API --> UI["Web App (1-to-1 Verify & Catalog Search)"]
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

## Verification Decision Logic & Lookalike Rejection

```mermaid
flowchart TD
    Pair["Input Pair: Reference Slab A vs Query Slab B"] --> TypeCheck{"Physical Relationship"}
    
    TypeCheck -->|Exact Same Physical Piece| SameP["Identical Micro-Vein Coordinates"]
    TypeCheck -->|Sibling Slabs (Same Quarry Block)| LookalikeP["Shared Base Color & Texture<br/>(Different Vein Fingerprints)"]
    TypeCheck -->|Different Stone Variety| DiffP["Distinct Color & Minerals"]

    SameP --> HomographyPass["Geometric Consensus: Homography Valid<br/>Inliers ≥ 35 | Inlier Ratio ≥ 30%"]
    HomographyPass --> SameVerdict["VERDICT: SAME PHYSICAL SLAB<br/>Confidence: 60.0% – 100.0%"]

    LookalikeP --> HomographyFail["Geometric Consensus: Homography Fails<br/>Inliers < 10 (Max 6 inliers observed)"]
    HomographyFail --> LookalikeVerdict["VERDICT: DIFFERENT PHYSICAL PIECE<br/>Confidence: 0.0%<br/>(Quarry Sibling Lookalike Correctly Rejected)"]

    DiffP --> DiffFail["Zero Feature Consistency<br/>Inliers < 4"]
    DiffFail --> DiffVerdict["VERDICT: DIFFERENT PHYSICAL PIECE<br/>Confidence: 0.0%"]

    classDef success fill:#d1e7dd,stroke:#0f5132,stroke-width:1.5px,color:#0f5132;
    classDef danger fill:#f8d7da,stroke:#842029,stroke-width:1.5px,color:#842029;
    class SameVerdict success;
    class LookalikeVerdict,DiffVerdict danger;
```

---

## Pipeline Engineering & Mathematical Foundations

### 1. Image Ingestion & CLAHE Ablation
- **Aspect-Ratio-Preserving Resize:** Longest edge constrained to $\le 1024\text{ px}$. Retains fine fissure resolution while capping pairwise latency to $< 250\text{ ms}$.
- **Grayscale Luminance Conversion:** Converts standard 3-channel BGR images to single-channel 8-bit grayscale ($Y = 0.299R + 0.587G + 0.114B$) to isolate edge gradients from illumination color casts.
- **CLAHE Ablation Decision:** Evaluated Contrast-Limited Adaptive Histogram Equalization. On speckled granites, CLAHE boosted surface grain noise by $300\%$ and increased nearest-neighbor matching time by $180\%$ without improving lookalike discrimination. **CLAHE is disabled by default** (`use_clahe=False`).

### 2. Feature Detection: Low-Threshold SIFT
- Standard OpenCV SIFT relies on `contrastThreshold = 0.04`, filtering out subtle veins on polished white Carrara or Statuario marble ($< 80$ keypoints).
- We tuned `contrastThreshold = 0.01` with `edgeThreshold = 10.0`, yielding $3,500\text{--}4,000$ keypoints on smooth marble while capturing faint vein branches.
- `n_features = 6000` prevents combinatorial runtime explosion on heavily speckled granite slabs.

### 3. Hellinger Kernel: RootSIFT Formulation
Standard SIFT uses Euclidean distance ($L_2$) on 128-dimensional gradient orientation histograms. $L_2$ implicitly models descriptor distances with a Gaussian distribution, giving disproportionate weight to large histogram bins.

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

Empirical testing across 100 pairs demonstrated that RootSIFT widens the separation gap between the weakest genuine match and the strongest lookalike by **$+13$ inliers ($+4.9\%$)**.

### 4. Ambiguity Filtering: Lowe's Ratio Test
For each descriptor $d_A$, let $n_{B,1}$ and $n_{B,2}$ be the first and second nearest neighbors with distances $r_1 \le r_2$:
$$\text{Valid Match} \iff \frac{r_1}{r_2} \le \tau_{\text{ratio}} \quad (\tau_{\text{ratio}} = 0.75)$$
Prunes repetitive crystalline patterns where ambiguous neighboring crystals produce nearly equidistant candidate matches.

### 5. Planar Homography: USAC_MAGSAC Consensus
Because stone slabs are rigid planar surfaces, any two views of the same slab obey a projective planar homography $H \in \mathbb{R}^{3 \times 3}$:
$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

We estimate $H$ using `cv2.findHomography` with `cv2.USAC_MAGSAC` (reprojection error $\tau_{\text{reproj}} = 4.0\text{ px}$, confidence $= 0.999$, $\text{maxIters} = 3000$). MAGSAC++ provides robust marginalization over noise scale, outperforming standard RANSAC under oblique viewing angles.

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

Evaluated across all 100 pairwise comparisons in `pairs.csv`:

### Summary Metrics

| Metric | Target Specification | Measured Result | Margin / Status |
|:---|:---:|:---:|:---:|
| **Same Pairs Classified `SAME`** | $\ge 90.0\%$ | **10 / 10 (100.0%)** | $+10.0\%$ (Perfect Recall) |
| **Different Pairs Correctly Rejected** | $\ge 95.0\%$ | **90 / 90 (100.0%)** | $+5.0\%$ (Perfect Rejection) |
| **Lookalike False Acceptances** | **0** | **0 / 10 (0 FAR)** | **Zero False Acceptances** |
| **Weakest Genuine Match ($N_{\text{inliers}}$)** | $> 40$ inliers | **284 inliers** | $7.1\times$ above minimum threshold |
| **Strongest Lookalike Sibling ($N_{\text{inliers}}$)** | $< 15$ inliers | **6 inliers** | Safety Separation Gap = **$47\times$** |
| **Average Latency per Pair** | $< 400\text{ ms}$ | **225.8 ms** | $43.5\%$ under latency budget |

### Breakdown by Distortion Category

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
├── src/                               # Core computer vision pipeline modules
│   ├── config.py                      # Immutable pipeline configurations & thresholds
│   ├── preprocessor.py                # Aspect-ratio resizing, grayscale conversion, CLAHE
│   ├── extractor.py                   # SIFT, RootSIFT transformation & FeatureCache
│   ├── matcher.py                     # BFMatcher, Lowe ratio test, USAC_MAGSAC homography
│   ├── scoring.py                     # Confidence score computation & 3-band classification
│   ├── visualizer.py                  # Inlier match rendering & base64 export
│   └── image_matcher.py               # Unified ImageMatcher class & match_pair() API
├── docs/                              # Project specifications & architecture design
│   ├── BRD.md                         # Business Requirements Document & Success Metrics
│   ├── PRD.md                         # Product Requirements Document & Functional Specs
│   ├── ARCHITECTURE.md                # System Architecture, Math & Algorithmic Contracts
│   └── AGILE_ROADMAP.md               # Sprints 1-4 execution roadmap & user stories
├── tests/                             # Automated pytest test suites
│   ├── test_sprint1.py                # Preprocessing, RootSIFT & caching unit tests
│   ├── test_sprint2.py                # Geometric verification & homography tests
│   ├── test_sprint3.py                # Scoring engine & end-to-end matching tests
│   ├── test_sprint4.py                # Benchmark integrity & self-healing tests
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
├── server.py                          # FastAPI production server with UI mount
├── pytest.ini                         # Standardized test runner configuration
├── REPORT.md                          # Detailed engineering report & failure analysis
├── requirements.txt                   # Pinned production dependencies
└── README.md                          # Main project documentation
```

---

## Quickstart Guide

### 1. Installation & Environment Setup

Create and activate a virtual environment, then install dependencies:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Interactive Web Dashboard & REST Server

Start the integrated web application (Backend REST API + Frontend UI):

```bash
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Once running, navigate to:
* **Web Dashboard:** [http://127.0.0.1:8000/ui/](http://127.0.0.1:8000/ui/) or [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
* **Interactive API Docs (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

**Web App Features:**
- **Tab 1: Verify Slab:** Upload two images or choose from organized presets (Genuine Same Pieces, Lookalikes, Identical Self-Matches). Supports mobile camera captures.
- **Tab 2: Catalog Search (1-to-N):** Upload an unlabelled photo to scan all registered references and rank matches by confidence.
- **Tab 3: Benchmark Audit:** Inspect live aggregate accuracy metrics and browse evaluated test pairs with filters.

### 3. Single-Pair Verification CLI (`match.py`)

Compare any two stone images from the terminal:

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

### 4. Batch Benchmark Evaluation CLI (`evaluate.py`)

Run the full 100-pair benchmark evaluation:

```bash
python evaluate.py \
  --data Intern_Assignment_Dataset \
  --pairs Intern_Assignment_Dataset/pairs.csv \
  --out results.csv
```

Outputs accuracy metrics to the terminal and records per-pair diagnostics into `results.csv`.

### 5. Parameter Ablation Study (`tune.py`)

Run the SIFT vs. RootSIFT empirical comparison across all benchmark pairs:

```bash
python tune.py --pairs Intern_Assignment_Dataset/pairs.csv --data Intern_Assignment_Dataset
```

---

## REST API Reference

The FastAPI service exposes standard REST endpoints:

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/` | Redirects to `/ui/` web dashboard. |
| `GET` | `/api/health` | Service health status check (`{"status": "ok"}`). |
| `GET` | `/api/presets` | List registered reference inventory and categorized queries. |
| `POST` | `/api/match` | Compare Image A and Image B (file uploads or presets). Auto-matches against inventory if only one image is supplied. Returns confidence, band, inliers, and lookalike diagnostics. |
| `POST` | `/api/gallery-search` | 1-to-N catalog search: ranks all inventory references against a query image. |
| `GET` | `/api/benchmark-results` | Returns aggregate benchmark accuracy statistics and evaluation rows. |

---

## Automated Test Suite

Run the full regression and integration test suite:

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
