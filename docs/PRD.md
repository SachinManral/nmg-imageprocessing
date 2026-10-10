# Product Requirements Document (PRD)
## NMG Labs — Image Matching Pipeline & Verification Engine

---

### Document Overview
- **Product Title:** NMG Stone Slab Geometric Verifier (`nmg-image-matching`)
- **Version:** 1.0.0
- **Phase:** AI Computer Vision Pipeline
- **Authors:** Advanced Computer Vision Agentic Team

---

### 1. Product Vision & Architecture Principles

The product is a deterministic, offline Python computer vision system designed to verify slab identities. It treats image matching as a strict physical object re-identification task:
1. **Fingerprint Verification:** Match unique physical keypoints on natural stone fissures and vein structures.
2. **Geometric Rigor:** Discard any feature cluster that does not conform to planar perspective transformation.
3. **Calibrated Confidence:** Disallow arbitrary arbitrary guesses; partition outcomes into high-conviction `SAME`, ambiguous `SIMILAR`, and negative `DIFFERENT`.
4. **Resilience & Defensibility:** Every decision must be mathematically defensible via inlier counts, inlier ratios, and spatial distribution diagnostics.

---

### 2. Functional Requirements (FR)

The system adheres strictly to the six-stage computer vision pipeline mandated by the candidate brief:

```
[ Input Images A & B ]
         │
         ▼
[ Stage 1: Preprocessing & Validation ]
   • Load & verify integrity
   • Resize to max dimension (e.g. 1024px)
   • Grayscale conversion & optional CLAHE
         │
         ▼
[ Stage 2: Feature Extraction ]
   • SIFT keypoint detection
   • RootSIFT descriptor transformation
         │
         ▼
[ Stage 3: Feature Matching & Ratio Test ]
   • Brute-Force L2 KNN matching (k=2)
   • Lowe's ratio test (threshold ~ 0.75)
         │
         ▼
[ Stage 4: Geometric Verification & Sanity ]
   • Homography via cv2.findHomography (USAC_MAGSAC / RANSAC)
   • Inlier mask extraction
   • Spatial spread / convex hull validation
   • Degeneracy & condition check
         │
         ▼
[ Stage 5: Similarity Scoring & Band Decision ]
   • Inlier-based score f(inliers, ratio, coverage) ∈ [0, 100]
   • Assign to: SAME / SIMILAR / DIFFERENT
         │
         ▼
[ Stage 6: Outputs & Artifact Generation ]
   • Inlier-only side-by-side visualization
   • CLI stdout metrics & CSV records
```

---

#### 2.1 FR-1: Image Ingestion & Preprocessing
- **FR-1.1 File Ingestion:** Accept standard image formats (`.jpg`, `.jpeg`, `.png`). If an image file is missing, corrupt, or unreadable, the system must log an error and exit gracefully without an unhandled stack trace.
- **FR-1.2 Image Resizing:** Resize images maintaining original aspect ratio such that the longest dimension does not exceed $L_{\max} = 1024\text{ px}$. This bounds descriptor extraction latency while preserving fine vein features.
- **FR-1.3 Color Space:** Convert BGR images to single-channel 8-bit grayscale ($Y = 0.299R + 0.587G + 0.114B$).
- **FR-1.4 Contrast Enhancement Evaluation (CLAHE):**
  - Implement Contrast Limited Adaptive Histogram Equalization (`cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`).
  - **Rule:** Do NOT apply CLAHE by default unless empirical ablation demonstrates positive impact on lookalike separation.

---

#### 2.2 FR-2: Feature Extraction (SIFT & RootSIFT)
- **FR-2.1 Baseline Detector:** SIFT (`cv2.SIFT_create(nfeatures=0, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)`).
- **FR-2.2 RootSIFT Formulation:**
  - Standard SIFT descriptors $D \in \mathbb{R}^{N \times 128}$ are matched using Euclidean distance, which corresponds to histogram intersection under a Gaussian noise model.
  - Apply the Hellinger kernel transformation:
    $$D_{L1} = \frac{D}{\|D\|_1 + \epsilon}$$
    $$D_{\text{root}} = \sqrt{D_{L1}}$$
  - Descendant property: $D_{\text{root}}$ already has unit $L_2$ norm (up to floating-point precision). Additional $L_2$ normalization is redundant.
- **FR-2.3 Diagnostics:** Track and report total keypoints detected: $K_A = |kp_A|$ and $K_B = |kp_B|$.

---

#### 2.3 FR-3: Feature Matching & Ambiguity Filtering
- **FR-3.1 Matcher:** `cv2.BFMatcher(cv2.NORM_L2)` executing $k$-Nearest Neighbors ($k=2$).
- **FR-3.2 Lowe's Ratio Test:**
  - Reject ambiguous matches where the distance to the closest neighbor $d_1$ is close to the second neighbor $d_2$:
    $$\text{match is valid} \iff \frac{d_1}{d_2} < \tau_{\text{ratio}} \quad (\text{default: } 0.75)$$
  - Track count of good matches: $M_{\text{good}}$.

---

#### 2.4 FR-4: Geometric Verification & Spatial Sanity Checks
- **FR-4.1 Homography Estimation:**
  - If $M_{\text{good}} < 4$, bypass matrix estimation, set inliers = 0, and output `DIFFERENT` (score = 0).
  - Compute homography matrix $H \in \mathbb{R}^{3 \times 3}$ using robust RANSAC (`cv2.USAC_MAGSAC` or `cv2.RANSAC`) with reprojection threshold $\tau_{\text{reproj}} = 4.0\text{ px}$.
- **FR-4.2 Inlier Extraction:**
  - Extract inlier count $N_{\text{inliers}}$ and compute inlier ratio:
    $$R_{\text{inlier}} = \frac{N_{\text{inliers}}}{M_{\text{good}}}$$
- **FR-4.3 Spatial Sanity Checks:**
  - Reject degenerate models where inliers collapse into a tiny cluster or collinear line.
  - Calculate normalized bounding box or convex hull area of inliers in both images:
    $$\text{Coverage}_A = \frac{\text{Area}(\text{ConvexHull}(pts_{A,\text{inliers}}))}{\text{Width}_A \times \text{Height}_A}$$
  - If inliers cover $< 2\%$ of the image area or inlier spread is singular, mark geometry as unverified.

---

#### 2.5 FR-5: Similarity Scoring & Decision Triad
- **FR-5.1 Confidence Score Formula:**
  - The score $S \in [0, 100]$ is a monotonically increasing function of geometric consensus:
    $$S = 0 \quad \text{if } H \text{ is invalid or } N_{\text{inliers}} < 4$$
    $$S = 100 \times \left( \alpha \cdot \min\left(1.0, \frac{N_{\text{inliers}}}{N_{\text{target}}}\right) + (1 - \alpha) \cdot R_{\text{inlier}} \right) \times \psi_{\text{spatial}}$$
    where $N_{\text{target}} \approx 60$, $\alpha \approx 0.65$, and $\psi_{\text{spatial}}$ penalizes localized point clustering.
- **FR-5.2 Three Decision Bands:**
  - `SAME`: High inlier count and high ratio ($N_{\text{inliers}} \ge 35$, $R_{\text{inlier}} \ge 0.35$, $S \ge 65\%$). Definite proof of identical physical object.
  - `SIMILAR`: Moderate inliers or partial consensus ($12 \le N_{\text{inliers}} < 35$, or low coverage due to heavy occlusion). Treated operationally as **not proven**.
  - `DIFFERENT`: Inliers near noise floor ($N_{\text{inliers}} < 12$, or $S < 30\%$).
- **FR-5.3 The Golden Rule:** Lookalike pairs (different slab cut from the same quarry block) must **NEVER** be classified as `SAME`.

---

#### 2.6 FR-6: Outputs, CLI & Artifacts
- **FR-6.1 Single Pair CLI (`match.py`):**
  ```bash
  python match.py --image-a data/images/img_001.jpg --image-b data/images/img_002.jpg --out output/
  ```
  - Stdout output:
    ```
    Image A keypoints: 4120
    Image B keypoints: 3890
    Total matches: 2840
    Good matches: 215
    RANSAC inliers: 142
    Inlier ratio: 0.66
    Confidence: 88.5%
    Decision band: SAME
    Processing time: 142 ms
    ```
  - Output artifact: `output/img_001__img_002.jpg` rendering side-by-side images with lines drawn **only for RANSAC inlier matches**.
- **FR-6.2 Batch Evaluation CLI (`evaluate.py`):**
  ```bash
  python evaluate.py --data data --pairs data/pairs.csv --out results.csv
  ```
  - Generates `results.csv` with exact columns:
    `image_a,image_b,label,pair_type,keypoints_a,keypoints_b,total_matches,good_matches,inliers,inlier_ratio,confidence,band,time_ms`
  - Handles hidden evaluation datasets gracefully: if `label` or `pair_type` are `"unknown"`, it writes all rows to `results.csv` and suppresses ground-truth metric calculations without crashing.
  - Prints accuracy summary tables (Same accuracy, Different accuracy, Lookalike false acceptance count, pair-type breakdown).

---

### 3. Non-Functional Requirements (NFR)

- **NFR-1 Determinism:** Zero stochastic variance across runs given identical inputs. Seed any random state in RANSAC algorithms if supported (`cv2.setRNGSeed`).
- **NFR-2 Latency:** $< 350\text{ ms}$ average latency per image pair at $1024\text{ px}$ resolution.
- **NFR-3 Memory Footprint:** Peak RSS $< 500\text{ MB}$. Release intermediate image arrays promptly.
- **NFR-4 Maintainability:** Clean separation of concerns between preprocessing, extraction, geometric solver, and CLI layers.
- **NFR-5 Extensibility:** Ready interface to plug into future FastAPI / Streamlit / React frontends.
