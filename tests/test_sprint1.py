"""
Tests for image preprocessing, SIFT/RootSIFT extraction, and caching.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import FeatureConfig, PreprocessConfig
from src.preprocessor import ImagePreprocessor
from src.extractor import FeatureCache, FeatureExtractor


SAMPLE_REF_IMG = "Intern_Assignment_Dataset/reference/img_036__B01A_carrara_reference.jpg"
NON_EXISTENT_IMG = "Intern_Assignment_Dataset/reference/does_not_exist.jpg"


def test_missing_image_graceful_handling():
    preprocessor = ImagePreprocessor()
    result = preprocessor.load_and_preprocess(NON_EXISTENT_IMG)

    assert not result.is_valid
    assert result.error_message is not None
    assert "File not found" in result.error_message
    assert result.gray.size == 0


def test_image_preprocessing_resize_and_aspect_ratio():
    assert Path(SAMPLE_REF_IMG).exists(), f"Sample test image not found: {SAMPLE_REF_IMG}"

    config = PreprocessConfig(max_dimension=1024, use_clahe=False)
    preprocessor = ImagePreprocessor(config)
    result = preprocessor.load_and_preprocess(SAMPLE_REF_IMG)

    assert result.is_valid
    assert result.error_message is None
    orig_h, orig_w = result.original_shape
    res_h, res_w = result.resized_shape

    assert max(res_h, res_w) <= 1024
    # Aspect ratio verification (within 1 pixel rounding error)
    orig_aspect = orig_w / float(orig_h)
    resized_aspect = res_w / float(res_h)
    assert abs(orig_aspect - resized_aspect) < 0.02
    assert len(result.gray.shape) == 2  # Single-channel grayscale


def test_image_preprocessing_with_clahe():
    config_clahe = PreprocessConfig(max_dimension=1024, use_clahe=True, clahe_clip_limit=2.0)
    preprocessor = ImagePreprocessor(config_clahe)
    result = preprocessor.load_and_preprocess(SAMPLE_REF_IMG)

    assert result.is_valid
    assert result.gray.dtype == np.uint8


def test_sift_feature_extraction():
    preprocessor = ImagePreprocessor()
    img = preprocessor.load_and_preprocess(SAMPLE_REF_IMG)

    extractor = FeatureExtractor(FeatureConfig(method="sift"))
    features = extractor.extract(img)

    assert features.num_keypoints > 0
    assert features.descriptors is not None
    assert features.descriptors.shape[1] == 128
    assert features.descriptors.dtype == np.float32


def test_rootsift_transformation_and_unit_l2_norm():
    preprocessor = ImagePreprocessor()
    img = preprocessor.load_and_preprocess(SAMPLE_REF_IMG)

    extractor = FeatureExtractor(FeatureConfig(method="rootsift"))
    features = extractor.extract(img)

    assert features.num_keypoints > 0
    assert features.descriptors is not None
    assert features.descriptors.shape[1] == 128

    # All RootSIFT values must be non-negative
    assert np.all(features.descriptors >= 0.0)

    # Inherent unit L2 norm verification: sum(z^2) == 1
    row_l2_norms = np.linalg.norm(features.descriptors, axis=1)
    # Average L2 norm should be extremely close to 1.0 (apart from slight float32 precision)
    np.testing.assert_allclose(row_l2_norms, 1.0, atol=1e-3)


def test_feature_cache_acceleration():
    preprocessor = ImagePreprocessor()
    img = preprocessor.load_and_preprocess(SAMPLE_REF_IMG)

    cache = FeatureCache()
    extractor = FeatureExtractor(FeatureConfig(method="rootsift"), cache=cache)

    assert cache.size() == 0

    # First call - populates cache
    features1 = extractor.extract(img)
    assert cache.size() == 1

    # Second call - retrieves from cache
    features2 = extractor.extract(img)
    assert cache.size() == 1
    assert features1.num_keypoints == features2.num_keypoints
    np.testing.assert_array_equal(features1.descriptors, features2.descriptors)

    # Clear cache
    cache.clear()
    assert cache.size() == 0


if __name__ == "__main__":
    test_missing_image_graceful_handling()
    test_image_preprocessing_resize_and_aspect_ratio()
    test_image_preprocessing_with_clahe()
    test_sift_feature_extraction()
    test_rootsift_transformation_and_unit_l2_norm()
    test_feature_cache_acceleration()
    print("All Sprint 1 test suites passed successfully!")
