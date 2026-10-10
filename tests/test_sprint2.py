import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FeatureConfig, MatchingConfig, PreprocessConfig, RANSACConfig
from src.extractor import FeatureExtractor
from src.matcher import FeatureMatcher
from src.preprocessor import ImagePreprocessor


REF_CARRARA = "Intern_Assignment_Dataset/reference/img_036__B01A_carrara_reference.jpg"
QUERY_SAME = "Intern_Assignment_Dataset/queries/img_021__B01A_rotation_scale_same.jpg"
QUERY_LOOKALIKE = "Intern_Assignment_Dataset/queries/img_013__B01B_viewpoint_lookalike.jpg"
REF_NERO = "Intern_Assignment_Dataset/reference/img_011__B04A_nero_reference.jpg"


def test_matching_same_slab_pair():
    preprocessor = ImagePreprocessor(PreprocessConfig(max_dimension=1024))
    extractor = FeatureExtractor(FeatureConfig(method="rootsift"))
    matcher = FeatureMatcher(MatchingConfig(ratio_threshold=0.75), RANSACConfig())

    img_a = preprocessor.load_and_preprocess(REF_CARRARA)
    img_b = preprocessor.load_and_preprocess(QUERY_SAME)

    feat_a = extractor.extract(img_a)
    feat_b = extractor.extract(img_b)

    out = matcher.match_features(feat_a, feat_b)

    assert out.good_matches > 20
    assert out.inliers >= 35
    assert out.inlier_ratio >= 0.30
    assert out.is_geometry_valid is True
    assert out.homography is not None
    assert out.homography.shape == (3, 3)
    assert out.spatial_coverage > 0.05


def test_matching_lookalike_slab_pair():
    preprocessor = ImagePreprocessor(PreprocessConfig(max_dimension=1024))
    extractor = FeatureExtractor(FeatureConfig(method="rootsift"))
    matcher = FeatureMatcher(MatchingConfig(ratio_threshold=0.75), RANSACConfig())

    img_a = preprocessor.load_and_preprocess(REF_CARRARA)
    img_b = preprocessor.load_and_preprocess(QUERY_LOOKALIKE)

    feat_a = extractor.extract(img_a)
    feat_b = extractor.extract(img_b)

    out = matcher.match_features(feat_a, feat_b)

    # Lookalike sibling from the same block must have low consensus inliers
    assert out.inliers < 20


def test_matching_different_stone_types():
    preprocessor = ImagePreprocessor(PreprocessConfig(max_dimension=1024))
    extractor = FeatureExtractor(FeatureConfig(method="rootsift"))
    matcher = FeatureMatcher(MatchingConfig(ratio_threshold=0.75), RANSACConfig())

    img_a = preprocessor.load_and_preprocess(REF_CARRARA)
    img_b = preprocessor.load_and_preprocess(REF_NERO)

    feat_a = extractor.extract(img_a)
    feat_b = extractor.extract(img_b)

    out = matcher.match_features(feat_a, feat_b)

    assert out.inliers < 10


def test_insufficient_matches_graceful_handling():
    matcher = FeatureMatcher()
    output = matcher._empty_match_output(total_matches=2, good_matches=1)

    assert output.inliers == 0
    assert output.is_geometry_valid is False
    assert output.homography is None
    assert output.spatial_coverage == 0.0


if __name__ == "__main__":
    test_matching_same_slab_pair()
    test_matching_lookalike_slab_pair()
    test_matching_different_stone_types()
    test_insufficient_matches_graceful_handling()
    print("All Sprint 2 test suites passed successfully!")
