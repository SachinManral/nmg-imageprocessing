"""
Pipeline configuration classes and default thresholds.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class PreprocessConfig:
    """Settings for image loading, resizing and enhancement."""
    max_dimension: int = 1024
    use_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)


@dataclass(frozen=True)
class FeatureConfig:
    """Settings for keypoint detection and descriptor extraction."""
    method: str = "rootsift"  # 'sift' or 'rootsift'
    n_features: int = 0       # 0 means retain all detected keypoints
    contrast_threshold: float = 0.04
    edge_threshold: float = 10.0
    sigma: float = 1.6


@dataclass(frozen=True)
class MatchingConfig:
    """Settings for descriptor matching and ambiguous match filtering."""
    matcher_type: str = "BF"  # 'BF' or 'FLANN'
    ratio_threshold: float = 0.75


@dataclass(frozen=True)
class RANSACConfig:
    """Settings for planar homography estimation and geometric sanity checks."""
    method: str = "USAC_MAGSAC"  # 'USAC_MAGSAC', 'RANSAC', or 'RHO'
    reproj_threshold: float = 4.0
    max_iters: int = 3000
    confidence: float = 0.999
    min_inliers_for_geometry: int = 8
    min_spatial_coverage: float = 0.02


@dataclass(frozen=True)
class ThresholdConfig:
    """Calibrated thresholds for the three decision bands."""
    same_min_inliers: int = 35
    same_min_ratio: float = 0.30
    same_min_confidence: float = 60.0
    similar_min_inliers: int = 10
    similar_min_confidence: float = 25.0


@dataclass(frozen=True)
class PipelineConfig:
    """Aggregated pipeline configuration."""
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    ransac: RANSACConfig = field(default_factory=RANSACConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
