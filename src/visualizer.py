from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np


class MatchVisualizer:
    @staticmethod
    def draw_and_save_inliers(
        image_a_bgr: np.ndarray,
        keypoints_a: List[cv2.KeyPoint],
        image_b_bgr: np.ndarray,
        keypoints_b: List[cv2.KeyPoint],
        inlier_matches: List[cv2.DMatch],
        output_path: str,
    ) -> bool:
        if image_a_bgr.size == 0 or image_b_bgr.size == 0:
            return False

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Draw only RANSAC inlier matches (green lines, single points omitted)
        vis_image = cv2.drawMatches(
            image_a_bgr,
            keypoints_a,
            image_b_bgr,
            keypoints_b,
            inlier_matches,
            None,
            matchColor=(0, 220, 0),
            singlePointColor=None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        # Windows-safe file write
        try:
            success, encoded_img = cv2.imencode(".jpg", vis_image)
            if success:
                encoded_img.tofile(str(out_file))
                return True
        except Exception:
            return False

        return False
