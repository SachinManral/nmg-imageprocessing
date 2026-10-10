import base64
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

        try:
            success, encoded_img = cv2.imencode(".jpg", vis_image)
            if success:
                encoded_img.tofile(str(out_file))
                return True
        except Exception:
            return False

        return False

    @staticmethod
    def render_inliers_to_base64(
        image_a_bgr: np.ndarray,
        keypoints_a: List[cv2.KeyPoint],
        image_b_bgr: np.ndarray,
        keypoints_b: List[cv2.KeyPoint],
        inlier_matches: List[cv2.DMatch],
        band: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> str:
        if image_a_bgr.size == 0 or image_b_bgr.size == 0:
            return ""

        vis_image = cv2.drawMatches(
            image_a_bgr,
            keypoints_a,
            image_b_bgr,
            keypoints_b,
            inlier_matches,
            None,
            matchColor=(0, 200, 0),
            singlePointColor=None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        # Optional top status header
        if band is not None and confidence is not None:
            h, w = vis_image.shape[:2]
            header_h = 44
            canvas = np.full((h + header_h, w, 3), 255, dtype=np.uint8)
            canvas[header_h:, :] = vis_image

            # Banner color by decision
            if band == "SAME":
                bg_color = (235, 250, 235)  # Light green
                text_color = (25, 135, 84)   # Forest green
                status_text = f"VERIFIED: SAME PHYSICAL SLAB ({confidence:.1f}% Confidence | {len(inlier_matches)} Inliers)"
            elif band == "SIMILAR":
                bg_color = (255, 248, 230)  # Light amber
                text_color = (180, 110, 0)   # Amber
                status_text = f"INCONCLUSIVE: SIMILAR PIECE ({confidence:.1f}% Confidence | {len(inlier_matches)} Inliers)"
            else:
                bg_color = (253, 237, 237)  # Light red
                text_color = (211, 47, 47)   # Deep red
                status_text = f"REJECTED: DIFFERENT PIECE (0.0% Confidence | No Consistent Geometry)"

            cv2.rectangle(canvas, (0, 0), (w, header_h), bg_color, -1)
            cv2.putText(
                canvas,
                status_text,
                (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                text_color,
                2,
                cv2.LINE_AA,
            )
            vis_image = canvas

        success, encoded = cv2.imencode(".jpg", vis_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            return ""

        b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
