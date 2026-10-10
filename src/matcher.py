from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.config import MatchingConfig, RANSACConfig
from src.extractor import ExtractedFeatures


@dataclass
class MatchOutput:
    total_matches: int
    good_matches: int
    good_match_objects: List[cv2.DMatch]
    inliers: int
    inlier_ratio: float
    inlier_matches: List[cv2.DMatch]
    homography: Optional[np.ndarray]
    inlier_mask: Optional[np.ndarray]
    spatial_coverage: float
    is_geometry_valid: bool


class FeatureMatcher:
    def __init__(
        self,
        matching_config: Optional[MatchingConfig] = None,
        ransac_config: Optional[RANSACConfig] = None,
    ):
        self.matching_config = matching_config or MatchingConfig()
        self.ransac_config = ransac_config or RANSACConfig()
        self._matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def match_features(
        self,
        features_a: ExtractedFeatures,
        features_b: ExtractedFeatures,
    ) -> MatchOutput:
        desc_a = features_a.descriptors
        desc_b = features_b.descriptors

        if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
            return self._empty_match_output()

        raw_matches = self._matcher.knnMatch(desc_a, desc_b, k=2)
        total_matches = len(raw_matches)

        # Lowe's ratio test to filter ambiguous descriptor matches
        ratio_thresh = self.matching_config.ratio_threshold
        good_matches: List[cv2.DMatch] = []
        for pair in raw_matches:
            if len(pair) == 2 and pair[0].distance < ratio_thresh * pair[1].distance:
                good_matches.append(pair[0])

        good_count = len(good_matches)
        if good_count < 4:
            return self._empty_match_output(total_matches=total_matches, good_matches=good_count)

        # Extract coordinates for homography estimation
        kp_a = features_a.keypoints
        kp_b = features_b.keypoints
        pts_a = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_b = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        method = self._get_ransac_method()
        try:
            homography, mask = cv2.findHomography(
                pts_a,
                pts_b,
                method,
                self.ransac_config.reproj_threshold,
                maxIters=self.ransac_config.max_iters,
                confidence=self.ransac_config.confidence,
            )
        except Exception:
            homography, mask = None, None

        if homography is None or mask is None:
            return self._empty_match_output(
                total_matches=total_matches,
                good_matches=good_count,
                good_match_objects=good_matches,
            )

        mask_flat = mask.ravel()
        num_inliers = int(np.sum(mask_flat == 1))
        inlier_ratio = num_inliers / float(good_count) if good_count > 0 else 0.0
        inlier_matches = [m for m, keep in zip(good_matches, mask_flat) if keep == 1]

        # Spatial sanity and dispersion check
        shape_a = features_a.processed_image.resized_shape
        shape_b = features_b.processed_image.resized_shape
        coverage, is_valid_geometry = self._validate_geometry(
            num_inliers=num_inliers,
            inlier_matches=inlier_matches,
            kp_a=kp_a,
            kp_b=kp_b,
            shape_a=shape_a,
            shape_b=shape_b,
        )

        return MatchOutput(
            total_matches=total_matches,
            good_matches=good_count,
            good_match_objects=good_matches,
            inliers=num_inliers,
            inlier_ratio=round(inlier_ratio, 4),
            inlier_matches=inlier_matches,
            homography=homography,
            inlier_mask=mask,
            spatial_coverage=round(coverage, 4),
            is_geometry_valid=is_valid_geometry,
        )

    def _get_ransac_method(self) -> int:
        name = self.ransac_config.method.upper()
        if name == "USAC_MAGSAC" and hasattr(cv2, "USAC_MAGSAC"):
            return cv2.USAC_MAGSAC
        if name == "RHO" and hasattr(cv2, "RHO"):
            return cv2.RHO
        return cv2.RANSAC

    def _validate_geometry(
        self,
        num_inliers: int,
        inlier_matches: List[cv2.DMatch],
        kp_a: List[cv2.KeyPoint],
        kp_b: List[cv2.KeyPoint],
        shape_a: Tuple[int, int],
        shape_b: Tuple[int, int],
    ) -> Tuple[float, bool]:
        if num_inliers < self.ransac_config.min_inliers_for_geometry:
            return 0.0, False

        h_a, w_a = shape_a
        h_b, w_b = shape_b
        if h_a <= 0 or w_a <= 0 or h_b <= 0 or w_b <= 0:
            return 0.0, False

        pts_inliers_a = np.float32([kp_a[m.queryIdx].pt for m in inlier_matches])
        pts_inliers_b = np.float32([kp_b[m.trainIdx].pt for m in inlier_matches])

        cov_a = self._compute_coverage(pts_inliers_a, w_a, h_a)
        cov_b = self._compute_coverage(pts_inliers_b, w_b, h_b)
        coverage = min(cov_a, cov_b)

        is_valid = coverage >= self.ransac_config.min_spatial_coverage
        return coverage, is_valid

    @staticmethod
    def _compute_coverage(points: np.ndarray, width: int, height: int) -> float:
        if len(points) < 3:
            return 0.0
        try:
            hull = cv2.convexHull(points)
            area = cv2.contourArea(hull)
            total_area = float(width * height)
            return area / total_area if total_area > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _empty_match_output(
        total_matches: int = 0,
        good_matches: int = 0,
        good_match_objects: Optional[List[cv2.DMatch]] = None,
    ) -> MatchOutput:
        return MatchOutput(
            total_matches=total_matches,
            good_matches=good_matches,
            good_match_objects=good_match_objects or [],
            inliers=0,
            inlier_ratio=0.0,
            inlier_matches=[],
            homography=None,
            inlier_mask=None,
            spatial_coverage=0.0,
            is_geometry_valid=False,
        )
