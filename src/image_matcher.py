from dataclasses import dataclass, field, replace
from pathlib import Path
import time
from typing import List, Optional
import cv2
import numpy as np

from src.config import PipelineConfig
from src.extractor import FeatureCache, FeatureExtractor
from src.matcher import FeatureMatcher
from src.preprocessor import ImagePreprocessor
from src.scoring import SimilarityScorer


@dataclass
class MatchResult:
    image_a: str
    image_b: str
    keypoints_a: int
    keypoints_b: int
    total_matches: int
    good_matches: int
    inliers: int
    inlier_ratio: float
    confidence: float
    band: str
    time_ms: float
    homography: Optional[np.ndarray] = None
    inlier_matches: List[cv2.DMatch] = field(default_factory=list)
    keypoints_a_objs: List[cv2.KeyPoint] = field(default_factory=list)
    keypoints_b_objs: List[cv2.KeyPoint] = field(default_factory=list)
    color_a: Optional[np.ndarray] = None
    color_b: Optional[np.ndarray] = None
    status: str = "SUCCESS"
    error_message: Optional[str] = None


class ImageMatcher:
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        cache: Optional[FeatureCache] = None,
    ):
        self.config = config or PipelineConfig()
        self.cache = cache
        self.preprocessor = ImagePreprocessor(self.config.preprocess)
        self.extractor = FeatureExtractor(self.config.feature, cache=self.cache)
        self.matcher = FeatureMatcher(self.config.matching, self.config.ransac)
        self.scorer = SimilarityScorer(self.config.thresholds)

    def match(self, image_a_path: str, image_b_path: str) -> MatchResult:
        start_time = time.perf_counter()

        # Run the normal pipeline first. It preserves the original image
        # contrast and is the lowest-cost option for most comparisons.
        result = self._match_once(
            image_a_path,
            image_b_path,
            self.preprocessor,
            self.extractor,
        )

        # Strong glare or uneven exposure can make the standard ratio test
        # discard real correspondences. For inconclusive pairs only, retry
        # with local contrast normalisation and retain it only when it offers
        # stronger geometric evidence. The same homography and coverage gates
        # still apply, so this is not a filename- or sample-specific rule.
        if (
            result.status == "SUCCESS"
            and result.band == "DIFFERENT"
            and self.config.preprocess.enable_clahe_fallback
            and not self.config.preprocess.use_clahe
        ):
            fallback_config = replace(self.config.preprocess, use_clahe=True)
            fallback_preprocessor = ImagePreprocessor(fallback_config)
            fallback_extractor = FeatureExtractor(self.config.feature, cache=self.cache)
            fallback_result = self._match_once(
                image_a_path,
                image_b_path,
                fallback_preprocessor,
                fallback_extractor,
            )
            if self._is_stronger_result(fallback_result, result):
                result = fallback_result

        result.time_ms = round((time.perf_counter() - start_time) * 1000.0, 1)
        return result

    def _match_once(
        self,
        image_a_path: str,
        image_b_path: str,
        preprocessor: ImagePreprocessor,
        extractor: FeatureExtractor,
    ) -> MatchResult:

        # Preprocessing
        img_a = preprocessor.load_and_preprocess(image_a_path)
        img_b = preprocessor.load_and_preprocess(image_b_path)

        if not img_a.is_valid:
            return self._error_result(image_a_path, image_b_path, img_a.error_message, 0.0)

        if not img_b.is_valid:
            return self._error_result(image_a_path, image_b_path, img_b.error_message, 0.0)

        # Feature extraction
        feat_a = extractor.extract(img_a)
        feat_b = extractor.extract(img_b)

        # Matching and geometric verification
        match_out = self.matcher.match_features(feat_a, feat_b)

        # Confidence scoring and band assignment
        score_out = self.scorer.compute_score(match_out)

        return MatchResult(
            image_a=image_a_path,
            image_b=image_b_path,
            keypoints_a=feat_a.num_keypoints,
            keypoints_b=feat_b.num_keypoints,
            total_matches=match_out.total_matches,
            good_matches=match_out.good_matches,
            inliers=match_out.inliers,
            inlier_ratio=match_out.inlier_ratio,
            confidence=score_out.confidence,
            band=score_out.band,
            time_ms=0.0,
            homography=match_out.homography,
            inlier_matches=match_out.inlier_matches,
            keypoints_a_objs=feat_a.keypoints,
            keypoints_b_objs=feat_b.keypoints,
            color_a=img_a.color,
            color_b=img_b.color,
            status="SUCCESS",
            error_message=None,
        )

    @staticmethod
    def _is_stronger_result(candidate: MatchResult, current: MatchResult) -> bool:
        """Prefer a fallback only when it improves the model's own evidence."""
        candidate_rank = {"DIFFERENT": 0, "SIMILAR": 1, "SAME": 2}
        current_rank = candidate_rank.get(current.band, 0)
        fallback_rank = candidate_rank.get(candidate.band, 0)
        if fallback_rank != current_rank:
            return fallback_rank > current_rank
        if candidate.confidence != current.confidence:
            return candidate.confidence > current.confidence
        return candidate.inliers > current.inliers

    @staticmethod
    def _error_result(
        img_a: str,
        img_b: str,
        msg: Optional[str],
        time_ms: float,
    ) -> MatchResult:
        return MatchResult(
            image_a=img_a,
            image_b=img_b,
            keypoints_a=0,
            keypoints_b=0,
            total_matches=0,
            good_matches=0,
            inliers=0,
            inlier_ratio=0.0,
            confidence=0.0,
            band="DIFFERENT",
            time_ms=round(time_ms, 1),
            status="ERROR",
            error_message=msg,
        )


def match_pair(
    image_a_path: str,
    image_b_path: str,
    config: Optional[PipelineConfig] = None,
    cache: Optional[FeatureCache] = None,
) -> MatchResult:
    """Convenience functional wrapper for pairwise matching."""
    matcher = ImageMatcher(config=config, cache=cache)
    return matcher.match(image_a_path, image_b_path)
