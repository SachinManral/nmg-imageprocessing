import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PipelineConfig, ThresholdConfig
from src.image_matcher import ImageMatcher, MatchResult, match_pair
from src.matcher import MatchOutput
from src.scoring import SimilarityScorer
import cv2
import numpy as np


REF_CARRARA = "Intern_Assignment_Dataset/reference/img_036__B01A_carrara_reference.jpg"
QUERY_SAME = "Intern_Assignment_Dataset/queries/img_021__B01A_rotation_scale_same.jpg"
QUERY_LOOKALIKE = "Intern_Assignment_Dataset/queries/img_013__B01B_viewpoint_lookalike.jpg"


def test_similarity_scorer_bands():
    scorer = SimilarityScorer(ThresholdConfig())

    # High consensus -> SAME
    high_match = MatchOutput(
        total_matches=500,
        good_matches=100,
        good_match_objects=[],
        inliers=80,
        inlier_ratio=0.80,
        inlier_matches=[],
        homography=None,
        inlier_mask=None,
        spatial_coverage=0.30,
        is_geometry_valid=True,
    )
    score_res = scorer.compute_score(high_match)
    assert score_res.band == "SAME"
    assert score_res.confidence >= 60.0

    # Intermediate evidence -> SIMILAR
    med_match = MatchOutput(
        total_matches=200,
        good_matches=50,
        good_match_objects=[],
        inliers=20,
        inlier_ratio=0.40,
        inlier_matches=[],
        homography=None,
        inlier_mask=None,
        spatial_coverage=0.15,
        is_geometry_valid=True,
    )
    score_res_med = scorer.compute_score(med_match)
    assert score_res_med.band == "SIMILAR"

    # Failed geometry -> DIFFERENT with 0.0 confidence
    failed_match = MatchOutput(
        total_matches=100,
        good_matches=20,
        good_match_objects=[],
        inliers=5,
        inlier_ratio=0.25,
        inlier_matches=[],
        homography=None,
        inlier_mask=None,
        spatial_coverage=0.01,
        is_geometry_valid=False,
    )
    score_res_fail = scorer.compute_score(failed_match)
    assert score_res_fail.band == "DIFFERENT"
    assert score_res_fail.confidence == 0.0


def test_match_pair_end_to_end():
    res_same = match_pair(REF_CARRARA, QUERY_SAME)
    assert res_same.status == "SUCCESS"
    assert res_same.band == "SAME"
    assert res_same.inliers >= 35

    res_lookalike = match_pair(REF_CARRARA, QUERY_LOOKALIKE)
    assert res_lookalike.status == "SUCCESS"
    assert res_lookalike.band == "DIFFERENT"


def test_clahe_fallback_keeps_only_stronger_geometric_result(monkeypatch):
    matcher = ImageMatcher(PipelineConfig())

    baseline = MatchResult(
        image_a="a.jpg", image_b="b.jpg", keypoints_a=10, keypoints_b=10,
        total_matches=10, good_matches=2, inliers=0, inlier_ratio=0.0,
        confidence=0.0, band="DIFFERENT", time_ms=0.0,
    )
    recovered = MatchResult(
        image_a="a.jpg", image_b="b.jpg", keypoints_a=10, keypoints_b=10,
        total_matches=10, good_matches=40, inliers=38, inlier_ratio=0.95,
        confidence=98.0, band="SAME", time_ms=0.0,
    )
    attempts = iter([baseline, recovered])
    monkeypatch.setattr(matcher, "_match_once", lambda *args: next(attempts))

    result = matcher.match("a.jpg", "b.jpg")

    assert result.band == "SAME"
    assert result.confidence == 98.0


def test_same_slab_survives_lighting_rotation_and_perspective_change(tmp_path):
    """Regression guard for real capture changes, without using file metadata."""
    source = cv2.imdecode(np.fromfile(REF_CARRARA, dtype=np.uint8), cv2.IMREAD_COLOR)
    height, width = source.shape[:2]

    # A projective transform models a shifted/tilted camera position. The
    # photometric adjustment models a darker, differently exposed capture.
    source_corners = np.float32([
        [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1],
    ])
    target_corners = np.float32([
        [80, 45], [width - 110, 0], [width - 25, height - 65], [25, height - 5],
    ])
    transform = cv2.getPerspectiveTransform(source_corners, target_corners)
    altered = cv2.warpPerspective(
        source, transform, (width, height), borderMode=cv2.BORDER_REPLICATE
    )
    altered = cv2.convertScaleAbs(altered, alpha=0.62, beta=35)

    altered_path = tmp_path / "inspection_photo.jpg"
    encoded_ok, encoded_image = cv2.imencode(".jpg", altered)
    assert encoded_ok
    encoded_image.tofile(str(altered_path))

    result = match_pair(REF_CARRARA, str(altered_path))

    assert result.status == "SUCCESS"
    assert result.band == "SAME"
    assert result.inliers >= 35


if __name__ == "__main__":
    test_similarity_scorer_bands()
    test_match_pair_end_to_end()
    print("All Sprint 3 test suites passed successfully!")
