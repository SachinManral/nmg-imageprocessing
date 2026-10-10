from dataclasses import dataclass
from typing import Tuple

from src.config import ThresholdConfig
from src.matcher import MatchOutput


@dataclass
class ScoreResult:
    confidence: float
    band: str  # "SAME", "SIMILAR", or "DIFFERENT"


class SimilarityScorer:
    def __init__(self, config: ThresholdConfig):
        self.config = config

    def compute_score(self, match_output: MatchOutput) -> ScoreResult:
        if not match_output.is_geometry_valid or match_output.inliers < 4:
            return ScoreResult(confidence=0.0, band="DIFFERENT")

        num_inliers = match_output.inliers
        inlier_ratio = match_output.inlier_ratio
        coverage = match_output.spatial_coverage

        # Inlier count saturation: 50+ inliers gives full credit
        count_factor = min(1.0, num_inliers / 50.0)
        ratio_factor = inlier_ratio

        # Weighted combination of inlier volume and consistency
        raw_score = (0.65 * count_factor + 0.35 * ratio_factor) * 100.0

        # Penalize tightly clustered inliers
        if coverage < 0.05:
            penalty = max(0.3, coverage / 0.05)
            raw_score *= penalty

        confidence = round(min(100.0, max(0.0, raw_score)), 1)
        band = self._classify_band(confidence, num_inliers, inlier_ratio)

        return ScoreResult(confidence=confidence, band=band)

    def _classify_band(
        self,
        confidence: float,
        num_inliers: int,
        inlier_ratio: float,
    ) -> str:
        # Strict criteria for SAME to prevent lookalike false acceptance
        if (
            confidence >= self.config.same_min_confidence
            and num_inliers >= self.config.same_min_inliers
            and inlier_ratio >= self.config.same_min_ratio
        ):
            return "SAME"

        if (
            confidence >= self.config.similar_min_confidence
            or num_inliers >= self.config.similar_min_inliers
        ):
            return "SIMILAR"

        return "DIFFERENT"
