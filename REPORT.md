# Engineering Report: Image Matching & Verification System

---

## 1. Algorithm Selection & Pipeline Design

Identifying whether two photographs depict the exact same physical stone slab requires verifying the unique geometric configuration of its mineral veins. Slabs cut consecutively from the same quarry block share the same color, texture, and mineral composition ("family resemblance"), but possess distinct vein trajectories ("fingerprints"). Global texture or color histogram methods fail on sibling lookalikes; our approach relies on local feature detection and planar geometric consensus.

### Preprocessing & Ingestion
- **Grayscale Conversion:** Preserves vein edge contrasts while removing illumination color casts.
- **Aspect-Ratio-Preserving Resize:** Longest edge bounded to $1024\text{ px}$. This maintains high spatial resolution for fine fissure detection while capping feature extraction runtime.
- **CLAHE Ablation:** We evaluated Contrast Limited Adaptive Histogram Equalization. While CLAHE boosted keypoint density on dark slabs, it artificially amplified surface grain noise in uniform granite and increased matching runtime by $180\%$ without improving lookalike separation. Consequently, CLAHE is disabled by default.

### Feature Detector: RootSIFT
- **Contrast Threshold Tuning:** Standard OpenCV SIFT uses `contrastThreshold = 0.04`, which filters out subtle veins on polished white marble (Carrara/Statuario yielded only 77 keypoints). Lowering the threshold to `0.01` increased keypoint yield to $3,500\text{--}4,000$ on marble slabs, revealing fine vein structures.
- **RootSIFT Transformation:** Standard SIFT descriptors $D$ undergo $L_1$ normalization followed by an element-wise square root: $z = \sqrt{D / \|D\|_1}$. Because $\sum (\sqrt{x})^2 = \sum x = 1$, the resulting descriptors naturally have unit $L_2$ norm, eliminating the need for redundant secondary normalization.
- **SIFT vs RootSIFT Empirical Benchmark (100 pairs):**
  - Standard SIFT: Same pair inliers min $272$, avg $1650.1$; Lookalike inliers min $4$, max $7$, avg $5.7$; separation margin $265$.
  - RootSIFT: Same pair inliers min $284$, avg $1685.4$; Lookalike inliers min $4$, max $6$, avg $4.6$; separation margin $278$.
  - RootSIFT widened the separation gap between the weakest genuine match and the strongest lookalike by $13$ inliers ($+4.9\%$), justifying its selection.
- **Keypoint Capping:** `n_features = 6000` prevents combinatorial explosion on heavily speckled granite slabs while capturing all vein features on marble.

### Matcher & Geometric Verification
- **Matcher:** Brute-Force matcher with $L_2$ distance.
- **Ambiguity Filter:** Lowe's ratio test at $\tau_{\text{ratio}} = 0.75$. This cleanly filters repetitive crystallization textures where the first and second nearest neighbors have similar descriptor distances.
- **Homography Estimation:** `cv2.findHomography` with `cv2.USAC_MAGSAC` and reprojection threshold $4.0\text{ px}$. MAGSAC++ provides robust marginalization over noise scale.
- **Spatial Coverage Validation:** We compute the convex hull area of inliers normalized by image dimensions. Homographies whose inliers collapse into tight clusters ($< 2\%$ image area) are rejected to eliminate accidental coincidences.

---

## 2. Confidence Scoring & Decision Bands

### Formula Rationale
The confidence score $S \in [0.0, 100.0]\%$ evaluates geometric consensus rather than raw match count. If geometric verification fails or inliers $< 4$, the score is strictly $0.0\%$.

When geometry is valid:
1. **Count Saturation:** $f_{\text{count}} = \min(1.0, N_{\text{inliers}} / 50.0)$
2. **Consistency Factor:** $f_{\text{ratio}} = N_{\text{inliers}} / M_{\text{good}}$
3. **Combined Base Score:** $S_{\text{raw}} = (0.65 \cdot f_{\text{count}} + 0.35 \cdot f_{\text{ratio}}) \times 100.0$
4. **Spatial Dispersion Dampener:** If convex hull coverage $< 5\%$, $S = S_{\text{raw}} \times \max(0.3, \text{coverage} / 0.05)$

### Decision Band Thresholds
- **`SAME`:** Confidence $\ge 60.0\%$, $N_{\text{inliers}} \ge 35$, and inlier ratio $\ge 0.30$.
- **`SIMILAR`:** Confidence $\ge 25.0\%$ or $N_{\text{inliers}} \ge 10$. Represents partial evidence (severe occlusion or extreme lighting) where identity cannot be conclusively proven.
- **`DIFFERENT`:** Inliers $< 10$ or failed geometric consensus.

---

## 3. Benchmark Results & Accuracy

Evaluated on the benchmark dataset across all 100 pairwise comparisons (5 reference slabs $\times$ 20 query slabs):

### Summary Metrics

| Metric | Target | Measured Result |
|---|---|---|
| Same pairs classified `SAME` | $\ge 90\%$ | **10 / 10 (100.0%)** |
| Different pairs correctly rejected | $\ge 95\%$ | **90 / 90 (100.0%)** |
| Lookalike pairs wrongly called `SAME` | **0** | **0 / 10 (0 False Acceptances)** |
| Weakest genuine match ($N_{\text{inliers}}$) | $> 40$ | **284 inliers** |
| Strongest lookalike sibling ($N_{\text{inliers}}$) | $< 15$ | **6 inliers** |
| Average latency per pair | $< 400\text{ ms}$ | **225.8 ms** |

### Breakdown by Pair Type

| Pair Type | Total Pairs | Ground Truth | Classified `SAME` | Classified `SIMILAR` | Classified `DIFFERENT` | Pass Rate |
|---|---|---|---|---|---|---|
| **rotation_scale** | 3 | Same | 3 | 0 | 0 | 100.0% |
| **viewpoint** | 3 | Same | 3 | 0 | 0 | 100.0% |
| **lighting** | 2 | Same | 2 | 0 | 0 | 100.0% |
| **occlusion** | 2 | Same | 2 | 0 | 0 | 100.0% |
| **lookalike** | 10 | Different | 0 | 0 | 10 | 100.0% |
| **different** | 80 | Different | 0 | 0 | 80 | 100.0% |

The separation between genuine matches (min 284 inliers) and sibling lookalikes (max 6 inliers) provides a safety margin of $47\times$.

---

## 4. Challenges Faced & Failure Mode Analysis

### Failure Mode 1: High-Key Polished Marble Keypoint Starvation
- **Observation:** In early iterations using standard SIFT (`contrastThreshold = 0.04`), Carrara marble queries produced fewer than 80 keypoints. Lowe's ratio test retained only 17 matches, causing legitimate matches under lighting shifts to fail geometric verification.
- **Root Cause:** Smooth crystalline limestone and white marble backgrounds have low intensity variance across gradient octaves.
- **Resolution:** Lowered SIFT contrast threshold to `0.01`. This captured subtle vein borders across Carrara and Statuario marble, raising keypoints to $\sim 3,500$ and inliers on same pairs to $> 1,200$.

### Failure Mode 2: Combinatorial Latency on Dense Granite
- **Observation:** Unbounded keypoint detection on grey granite (`img_003`) produced 17,857 keypoints. Performing exhaustive $k$-NN matching took over $4,200\text{ ms}$ per comparison.
- **Root Cause:** Mineral speckles in granite generate hundreds of localized corner extrema per square inch.
- **Resolution:** Introduced `n_features = 6000` upper bound and an in-memory `FeatureCache`. For batch evaluation, pre-computed descriptors are reused across queries, reducing average latency to $225.8\text{ ms}$ without altering matching decisions.

---

## 5. Potential Improvements with Additional Time

1. **Adaptive Contrast Normalization:** Automatically adjust `contrastThreshold` based on the standard deviation of local Laplacian energy, detecting low-contrast marble without over-detecting granular stone.
2. **Affine Invariant Local Adaptation (ASIFT):** Simulate camera viewpoint tilts prior to extraction to enhance tolerance under extreme $> 60^\circ$ oblique angles.
3. **Spatial Reranking with Delaunay Triangulation:** Construct local planar graphs between neighboring vein junctions to verify topology preservation in addition to global homography.
