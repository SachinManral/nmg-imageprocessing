"""
Image loading, aspect-ratio-preserving resize, and grayscale conversion.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from src.config import PreprocessConfig


@dataclass
class ProcessedImage:
    """Represents a preprocessed image with dimensional metadata."""
    path: str
    color: np.ndarray
    gray: np.ndarray
    original_shape: Tuple[int, int]  # (height, width)
    resized_shape: Tuple[int, int]   # (height, width)
    scale_factor: float
    use_clahe: bool = False
    is_valid: bool = True
    error_message: Optional[str] = None


class ImagePreprocessor:
    """Encapsulates image loading, aspect-ratio resizing, and enhancement."""

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        if self.config.use_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_grid_size,
            )
        else:
            self._clahe = None

    def load_and_preprocess(self, image_path: str) -> ProcessedImage:
        """
        Loads an image from disk and applies preprocessing steps.
        Guarantees no crashing on missing or corrupted image files.
        """
        path_obj = Path(image_path)
        if not path_obj.exists():
            return ProcessedImage(
                path=image_path,
                color=np.empty((0, 0, 3), dtype=np.uint8),
                gray=np.empty((0, 0), dtype=np.uint8),
                original_shape=(0, 0),
                resized_shape=(0, 0),
                scale_factor=1.0,
                is_valid=False,
                error_message=f"File not found: {image_path}",
            )

        # Windows-safe read supporting non-ASCII / space-containing paths
        try:
            raw_bytes = np.fromfile(str(path_obj), dtype=np.uint8)
            img_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            return ProcessedImage(
                path=image_path,
                color=np.empty((0, 0, 3), dtype=np.uint8),
                gray=np.empty((0, 0), dtype=np.uint8),
                original_shape=(0, 0),
                resized_shape=(0, 0),
                scale_factor=1.0,
                is_valid=False,
                error_message=f"Failed to read image bytes: {str(e)}",
            )

        if img_bgr is None or img_bgr.size == 0:
            return ProcessedImage(
                path=image_path,
                color=np.empty((0, 0, 3), dtype=np.uint8),
                gray=np.empty((0, 0), dtype=np.uint8),
                original_shape=(0, 0),
                resized_shape=(0, 0),
                scale_factor=1.0,
                is_valid=False,
                error_message=f"Corrupt or unreadable image format: {image_path}",
            )

        orig_h, orig_w = img_bgr.shape[:2]
        max_dim = self.config.max_dimension

        # Aspect-ratio-preserving resize
        longest_edge = max(orig_h, orig_w)
        if longest_edge > max_dim:
            scale_factor = max_dim / float(longest_edge)
            new_w = int(round(orig_w * scale_factor))
            new_h = int(round(orig_h * scale_factor))
            resized_bgr = cv2.resize(
                img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
        else:
            scale_factor = 1.0
            resized_bgr = img_bgr

        # Conversion to single-channel 8-bit grayscale
        gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)

        # Optional CLAHE enhancement
        if self._clahe is not None:
            gray = self._clahe.apply(gray)

        return ProcessedImage(
            path=image_path,
            color=resized_bgr,
            gray=gray,
            original_shape=(orig_h, orig_w),
            resized_shape=(gray.shape[0], gray.shape[1]),
            scale_factor=scale_factor,
            use_clahe=self.config.use_clahe,
            is_valid=True,
            error_message=None,
        )
