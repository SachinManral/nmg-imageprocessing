import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ThresholdConfig
from src.image_matcher import match_pair
from src.matcher import MatchOutput
from src.scoring import SimilarityScorer


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


if __name__ == "__main__":
    test_similarity_scorer_bands()
    test_match_pair_end_to_end()
    print("All Sprint 3 test suites passed successfully!")
