# Business Requirements Document (BRD)
## NMG Labs — Precision Stone Slab Image Matching & Verification System

---

### Document Control
- **Project Name:** NMG Labs Stone Image Matching Challenge (AI Pipeline Phase)
- **Document Version:** 1.0.0
- **Status:** Approved / Base Architecture
- **Target Audience:** Engineering Leads, Computer Vision Specialists, Product Managers, Technical Assessors

---

### 1. Executive Summary

Natural stone (e.g., Carrara marble, Statuario marble, Botticino, Nero granite) is an ultra-high-value building material where slabs quarried from the same block exhibit nearly identical base minerals, color casts, and general vein textures ("family resemblance"), yet possess distinct micro-veining topologies ("fingerprints"). 

In warehouse logistics, installation tracking, and insurance/custody verification, standard texture-matching or appearance-based computer vision models fail catastrophically: they confuse sibling slabs cut from the same block (**lookalikes**) with the **exact same physical piece**. 

The goal of this initiative is to build a robust, reproducible, offline Python computer vision pipeline that verifies whether two photos depict the **exact same physical stone slab** across severe photometric and geometric distortions (rotation, scale changes, oblique viewpoints, non-uniform glare, and up to 55% occlusion), while guaranteeing **zero false acceptances on lookalike slabs**.

---

### 2. Business Context & Problem Statement

#### 2.1 The Core Problem: "Fingerprint vs. Family Resemblance"
- **Same Physical Object:** The exact identical stone slab, re-imaged under varying environmental perturbations (different lighting, tilt angles, partial obstruction by tools/packing, scale changes).
- **Lookalike Piece:** A neighboring sibling slab sliced from the same quarry block. It shares identical chromaticity, texture density, and geological vein type, but its physical vein paths and surface micro-features differ.
- **Different Piece:** Slabs quarried from completely distinct stone blocks or varieties.

#### 2.2 Business Impact of Misclassification
1. **False Acceptance (Type I Error on Lookalikes):** Critical business failure. Mistaking a lookalike slab for the registered piece leads to mismatched inventory deliveries, invalid insurance claims, and compromised architectural installations where paired book-matching fails.
2. **False Rejection (Type II Error on Same Pieces):** Operational friction requiring secondary manual inspection. Acceptable under extreme occlusion or distortion, but must be minimized.
3. **Ambiguity Handling:** When evidence is inconclusive, the system must issue a calibrated `SIMILAR` verdict ("not proven") rather than guessing `SAME`.

---

### 3. Business Goals & Objectives

| Objective ID | Business Objective | Target Metric |
|---|---|---|
| **BO-1** | **Eliminate Lookalike False Acceptances** | **0% False Acceptance Rate (FAR)** on sibling lookalike pairs in benchmark sets. |
| **BO-2** | **High Same-Piece Recall** | $\ge 90\%$ True Acceptance Rate on same-piece pairs undergoing rotations, tilt, lighting shifts, and partial occlusions. |
| **BO-3** | **Calibrated Decision Triad** | Clean separation across 3 bands: `SAME` (proven), `SIMILAR` (unproven/ambiguous), `DIFFERENT` (no consistent geometry). |
| **BO-4** | **Air-Gapped Operational Independence** | 100% offline execution; zero external cloud API dependencies; strict adherence to open-source licensing. |
| **BO-5** | **Low-Latency Verification** | Processing time $< 400\text{ ms}$ per image pair on standard commodity hardware. |

---

### 4. Stakeholders & Personas

- **Warehouse & Inventory Auditor:** Requires automated, instant confirmation that a loaded slab matches the bill of lading.
- **Computer Vision Assessor / Client Evaluation Team:** Requires clean CLI interfaces (`match.py`, `evaluate.py`), deterministic outputs, and honest metric reporting with no hardcoded assumptions.
- **Quality Assurance & Verification Operator:** Relies on inlier-only overlay visualizations to inspect and defend matching decisions.

---

### 5. Scope of Work

#### 5.1 In-Scope (Phase 1 — AI Pipeline)
- Complete offline classical computer vision pipeline using OpenCV and NumPy.
- Preprocessing engine evaluating grayscale conversion, aspect-ratio-preserving resizing, and empirical validation of CLAHE.
- Scale- and rotation-invariant feature extraction (SIFT baseline and RootSIFT Hellinger kernel).
- Feature matching using $L_2$ distance with Lowe's ratio test.
- Robust geometric verification via homography estimation (RANSAC / MAGSAC++) and spatial sanity checks.
- Calibrated confidence scoring engine mapping inlier count, inlier ratio, and spatial coverage into 3 bands: `SAME`, `SIMILAR`, `DIFFERENT`.
- Strict command-line interfaces (`match.py` and batch `evaluate.py`) producing inlier-only visualization plots and CSV outputs.
- Comprehensive empirical report (`REPORT.md`) documenting performance, ablation studies, and genuine failure modes.

#### 5.2 Out-of-Scope (Deferred to Phase 2)
- Web graphical user interface (GUI) and web frontend dashboard.
- REST / WebSocket API server wrapper.
- Deep-learning fine-tuning or proprietary foundation models (SuperPoint/SuperGlue prohibited by non-commercial license).

---

### 6. Risks, Constraints & Mitigations

| Risk / Constraint | Business Impact | Mitigation Strategy |
|---|---|---|
| **Vein Similarity in Lookalikes** | High risk of spurious feature matches between sibling slabs cut inches apart in the same block. | Enforce Lowe's ratio test ($\le 0.75$) followed by strict RANSAC geometric consensus; spurious matches fail planar homography consistency. |
| **Severe Occlusion (25%–55%)** | Genuine matching points are masked, depressing total inlier count. | Combine raw inlier count with relative inlier ratio and bounding polygon area checks; fall back to `SIMILAR` rather than failing to `DIFFERENT`. |
| **Degenerate Homographies (Collinear / Clustered inliers)** | High inlier count clustered in a 20-pixel region can produce a mathematically valid but physically nonsensical homography. | Implement convex hull area tests, spatial dispersion thresholds, and eigenvalue/condition sanity checks on the transformation matrix. |
| **Evaluation Hidden Sets** | Evaluators run `evaluate.py` against secret unlabelled test sets where label columns contain "unknown". | Ensure `evaluate.py` safely skips metrics generation when labels are unknown while flawlessly writing `results.csv`. |

---

### 7. Success Criteria & Definition of Done

1. Both required CLI commands execute without error:
   - `python match.py --image-a <path> --image-b <path> --out <dir>`
   - `python evaluate.py --data <data_dir> --pairs <pairs.csv> --out results.csv`
2. Zero lookalike pairs classified as `SAME`.
3. Inlier-only matched image visualizations generated for verification.
4. Comprehensive empirical report with data-backed accuracy tables and failure analysis.
