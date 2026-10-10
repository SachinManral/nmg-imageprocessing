# Agile Project Plan & Sprint Breakdown
## NMG Labs — Image Matching Challenge & Enterprise Slab Verifier

---

### 1. Agile Framework & Execution Model

This project follows a streamlined **Scrum / Agile framework** organized into discrete, value-delivering sprints. Per project requirements, **Phase 1 strictly focuses on the core AI Computer Vision Pipeline**, while **Phase 2 encapsulates the Web Application & UI Frontend**.

```
┌────────────────────────────────────────────────────────────────────────┐
│                   PHASE 1: AI COMPUTER VISION PIPELINE                 │
├────────────────────┬────────────────────┬──────────────────────────────┤
│      SPRINT 1      │      SPRINT 2      │           SPRINT 3           │
│   Preprocessing    │  Matching, RANSAC  │      Scoring, Thresholds     │
│   & Feature Engine │  & Spatial Checks  │      & Official CLIs         │
├────────────────────┴────────────────────┴──────────────────────────────┤
│                                SPRINT 4                                │
│          Empirical Benchmarking, Ablations & REPORT.md                 │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               PHASE 2: ENTERPRISE UI & FRONTEND (FUTURE)               │
├────────────────────────────────────────────────────────────────────────┤
│                                SPRINT 5                                │
│       Interactive Web Visualizer, REST API & Batch Dashboard           │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 2. Sprint Backlog & User Stories (Phase 1: AI Pipeline)

#### Sprint 1: Foundation, Ingestion & Feature Extraction Engine
**Goal:** Deliver rock-solid image loading, aspect-ratio-preserving resizing, and feature extractors (SIFT + RootSIFT) with in-memory caching.

- **Story 1.1: Robust Image Preprocessor & Error Handling**
  - *As an* evaluation engineer,
  - *I want* images to be validated, converted to grayscale, and resized to $1024\text{ px}$ max edge,
  - *So that* memory is bounded and corrupted/missing files fail gracefully without unhandled exceptions.
  - *Acceptance Criteria:*
    - Missing or unreadable files return a clean `MatchResult(status="FILE_NOT_FOUND", band="DIFFERENT", confidence=0.0)`.
    - Aspect ratio is preserved upon resizing.
    - Grayscale conversion is standard single-channel uint8.

- **Story 1.2: SIFT & RootSIFT Extractor with Caching**
  - *As an* algorithm developer,
  - *I want* both standard SIFT and RootSIFT descriptors implemented,
  - *So that* we can test Hellinger distance vs Euclidean distance for marble grain patterns.
  - *Acceptance Criteria:*
    - RootSIFT applies $L_1$ normalization + square root.
    - No redundant second $L_2$ normalization is performed.
    - Keypoint counts for Image A and Image B are recorded and returned.
    - Extracted features are cached by image path to prevent redundant calculations during batch runs.

---

#### Sprint 2: Feature Matching, RANSAC & Spatial Sanity Checks
**Goal:** Deliver accurate $L_2$ nearest-neighbor matching, Lowe's ratio test, and robust geometric verification capable of handling rotations, viewpoints, lighting shifts, and occlusions.

- **Story 2.1: L2 Feature Matcher & Lowe's Ratio Test**
  - *As a* computer vision engineer,
  - *I want* BFMatcher with $L_2$ distance and Lowe's ratio test ($0.75$),
  - *So that* ambiguous and non-distinctive marble grain matches are filtered out.
  - *Acceptance Criteria:*
    - Distance metric is Euclidean ($L_2$) for SIFT/RootSIFT.
    - $k=2$ KNN matching implemented.
    - Ratio test drops matches where $d_1 / d_2 > 0.75$.
    - Returns `total_matches` and `good_matches`.

- **Story 2.2: Planar Homography & Robust RANSAC Estimation**
  - *As a* verification engineer,
  - *I want* homography estimated via `cv2.findHomography` with `cv2.USAC_MAGSAC` (or `cv2.RANSAC`),
  - *So that* only inlier matches that fit a physical planar transformation are retained.
  - *Acceptance Criteria:*
    - Pairs with $< 4$ good matches bypass RANSAC without crashing, returning 0 inliers.
    - Inlier mask correctly isolates true consensus matches.
    - `inlier_ratio` calculated as $N_{\text{inliers}} / M_{\text{good}}$.

- **Story 2.3: Spatial Dispersion & Degeneracy Verification**
  - *As a* quality lead,
  - *I want* inliers checked for spatial area coverage and non-collinearity,
  - *So that* accidental clusters or degenerate lines are not falsely accepted as matching objects.
  - *Acceptance Criteria:*
    - Compute convex hull area of inliers relative to image dimensions.
    - Reject homographies where inliers are clustered into an infinitesimal point or collinear line.

---

#### Sprint 3: Confidence Scoring, Decision Triad & Official CLIs
**Goal:** Implement the calibrated confidence score formula and the exact required command-line tools (`match.py` and `evaluate.py`).

- **Story 3.1: Confidence Scoring & Three Decision Bands**
  - *As an* operations auditor,
  - *I want* a 0–100% confidence score that places pairs into `SAME`, `SIMILAR`, or `DIFFERENT`,
  - *So that* lookalike pairs from the same quarry block are strictly rejected from `SAME`.
  - *Acceptance Criteria:*
    - Geometric failure yields exactly $0.0\%$ confidence.
    - Inlier count, inlier ratio, and spatial coverage form a calibrated score.
    - `SAME`: High inliers ($\ge 35$), high ratio ($\ge 0.30$), score $\ge 60\%$.
    - `SIMILAR`: Intermediate evidence ($10 \le \text{inliers} < 35$ or low coverage).
    - `DIFFERENT`: Inliers $< 10$ or failed geometry.
    - 0 lookalikes classified as `SAME`.

- **Story 3.2: Single-Pair CLI (`match.py`) & Inlier Visualization**
  - *As an* evaluation auditor,
  - *I want* `python match.py --image-a <A> --image-b <B> --out <out_dir>`,
  - *So that* I can inspect metrics on stdout and view side-by-side inlier-only drawings.
  - *Acceptance Criteria:*
    - Accepts `--image-a`, `--image-b`, and `--out`.
    - Prints all required fields: keypoints A/B, total matches, good matches, inliers, inlier ratio, confidence, band, time_ms.
    - Saves `<stem_a>__<stem_b>.jpg` rendering **inlier matches only**.

- **Story 3.3: Batch Evaluation CLI (`evaluate.py`) & Hidden-Set Tolerance**
  - *As an* assessment pipeline,
  - *I want* `python evaluate.py --data <data_dir> --pairs <pairs.csv> --out results.csv`,
  - *So that* batch evaluation runs seamlessly and outputs standard CSV records.
  - *Acceptance Criteria:*
    - Reads pairs CSV with headers: `image_a, image_b, label, pair_type`.
    - Handles hidden sets where `label` and `pair_type` are `"unknown"` without crashing.
    - Exports `results.csv` with exact 13 required columns.
    - Prints summary accuracy breakdown when labels are known.

---

#### Sprint 4: Empirical Benchmarking, Ablations & Technical Report
**Goal:** Run end-to-end evaluations on the 100-pair benchmark dataset, conduct CLAHE and RootSIFT ablation experiments, and generate `REPORT.md` and `README.md`.

- **Story 4.1: Dataset Exploration & Benchmark Execution**
  - *As a* research lead,
  - *I want* to run `evaluate.py` across all 100 benchmark comparisons (5 references $\times$ 20 queries),
  - *So that* we obtain real empirical metrics across rotation/scale, viewpoint, lighting, occlusion, and lookalike sibling pairs.
  - *Acceptance Criteria:*
    - 100% of benchmark pairs processed.
    - Full `results.csv` generated.

- **Story 4.2: Ablation Experiments (CLAHE & SIFT vs RootSIFT vs ORB)**
  - *As an* algorithm architect,
  - *I want* empirical proof of whether CLAHE and RootSIFT improve lookalike rejection margin,
  - *So that* design decisions in `REPORT.md` are backed by data rather than assumptions.
  - *Acceptance Criteria:*
    - Compare inlier separation on lookalikes vs genuine matches with and without CLAHE.
    - Compare RootSIFT vs standard SIFT.
    - Record findings in `REPORT.md`.

- **Story 4.3: Technical Deliverables (`REPORT.md`, `README.md`, `requirements.txt`)**
  - *As an* engineering candidate,
  - *I want* comprehensive documentation explaining our design choices, failure modes, and setup in $\le 3$ commands,
  - *So that* our submission is immediately reproducible and defendable.
  - *Acceptance Criteria:*
    - `README.md` with $\le 3$ execution commands.
    - `REPORT.md` with verified accuracy tables, 2 analyzed failure cases, and future enhancements.
    - `requirements.txt` with tested dependencies.

---

### 3. Sprint Velocity & Definition of Done (DoD)

#### Definition of Ready (DoR):
- Requirements clearly stated with unambiguous input/output signatures.
- Test data available and path verified.
- Open-source constraints adhered to.

#### Definition of Done (DoD):
1. Code passes all unit execution checks and adheres to Python PEP 8 standards.
2. CLI commands execute with exact expected parameter names.
3. Zero crashes on missing files or hidden unlabelled pairs.
4. Output visualization renders only inlier matches.
5. Zero lookalikes classified as `SAME`.
6. Code pushed to git repository with clean commit history.
