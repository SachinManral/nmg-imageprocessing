"""
Feature extraction using SIFT and RootSIFT with in-memory caching.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.config import FeatureConfig
from src.preprocessor import ProcessedImage


@dataclass
class ExtractedFeatures:
    """Contains detected keypoints and their corresponding descriptors."""
    keypoints: List[cv2.KeyPoint]
    descriptors: Optional[np.ndarray]  # (N, 128) float32
    num_keypoints: int
    method: str
    processed_image: ProcessedImage


class FeatureCache:
    """
    In-memory cache for extracted features.
    Keyed by file path and last modification timestamp to ensure freshness.
    Cache strictly optimizes execution speed; it has zero impact on output results.
    """

    def __init__(self):
        self._cache: Dict[Tuple[str, float, str, int, bool], ExtractedFeatures] = {}

    def make_key(
        self,
        image_path: str,
        method: str,
        max_dim: int,
        use_clahe: bool,
    ) -> Tuple[str, float, str, int, bool]:
        """Constructs an invariant cache key."""
        path_obj = Path(image_path)
        mtime = path_obj.stat().st_mtime if path_obj.exists() else 0.0
        return (str(path_obj.resolve()), mtime, method.lower(), max_dim, use_clahe)

    def get(
        self,
        image_path: str,
        method: str,
        max_dim: int,
        use_clahe: bool,
    ) -> Optional[ExtractedFeatures]:
        """Retrieves cached features if present."""
        key = self.make_key(image_path, method, max_dim, use_clahe)
        return self._cache.get(key)

    def put(
        self,
        image_path: str,
        method: str,
        max_dim: int,
        use_clahe: bool,
        features: ExtractedFeatures,
    ) -> None:
        """Stores extracted features in the cache."""
        key = self.make_key(image_path, method, max_dim, use_clahe)
        self._cache[key] = features

    def clear(self) -> None:
        """Flushes the feature cache."""
        self._cache.clear()

    def size(self) -> int:
        """Returns number of cached items."""
        return len(self._cache)


class FeatureExtractor:
    """Extracts scale- and rotation-invariant features using SIFT or RootSIFT."""

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        cache: Optional[FeatureCache] = None,
    ):
        self.config = config or FeatureConfig()
        self.cache = cache
        self._sift = cv2.SIFT_create(
            nfeatures=self.config.n_features,
            contrastThreshold=self.config.contrast_threshold,
            edgeThreshold=self.config.edge_threshold,
            sigma=self.config.sigma,
        )

    @staticmethod
    def compute_rootsift(descriptors: Optional[np.ndarray], eps: float = 1e-7) -> Optional[np.ndarray]:
        """
        Applies the Hellinger kernel transformation (RootSIFT):
          1. L1 normalization per descriptor row: x / (sum(|x|) + eps)
          2. Element-wise square root: sqrt(x_l1)
        Note: The resulting vectors inherently have unit L2 norm (sum of sqrt(x)^2 = sum(x) = 1).
        Additional L2 normalization is mathematically redundant and omitted.
        """
        if descriptors is None or descriptors.shape[0] == 0:
            return descriptors

        # Ensure float32 representation
        desc_float = descriptors.astype(np.float32, copy=False)

        # L1 normalization
        l1_norm = np.sum(np.abs(desc_float), axis=1, keepdims=True) + eps
        desc_l1 = desc_float / l1_norm

        # Element-wise square root
        rootsift_desc = np.sqrt(desc_l1)
        return rootsift_desc

    def extract(self, processed_image: ProcessedImage) -> ExtractedFeatures:
        """
        Detects keypoints and computes descriptors on a preprocessed image.
        Uses in-memory cache if available.
        """
        if not processed_image.is_valid or processed_image.gray.size == 0:
            return ExtractedFeatures(
                keypoints=[],
                descriptors=None,
                num_keypoints=0,
                method=self.config.method,
                processed_image=processed_image,
            )

        # Check cache if cache is provided
        if self.cache is not None:
            cached = self.cache.get(
                image_path=processed_image.path,
                method=self.config.method,
                max_dim=max(processed_image.resized_shape),
                use_clahe=processed_image.use_clahe,
            )
            if cached is not None:
                return cached

        # Detect and compute SIFT features
        keypoints, descriptors = self._sift.detectAndCompute(processed_image.gray, None)

        if descriptors is not None:
            if self.config.method.lower() == "rootsift":
                descriptors = self.compute_rootsift(descriptors)
            else:
                descriptors = descriptors.astype(np.float32, copy=False)
            num_kp = len(keypoints)
        else:
            keypoints = []
            descriptors = None
            num_kp = 0

        features = ExtractedFeatures(
            keypoints=keypoints,
            descriptors=descriptors,
            num_keypoints=num_kp,
            method=self.config.method,
            processed_image=processed_image,
        )

        # Store in cache
        if self.cache is not None:
            self.cache.put(
                image_path=processed_image.path,
                method=self.config.method,
                max_dim=max(processed_image.resized_shape),
                use_clahe=processed_image.use_clahe,
                features=features,
            )

        return features
