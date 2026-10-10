import argparse
from pathlib import Path
import sys

from src.image_matcher import match_pair
from src.visualizer import MatchVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two stone slab images.")
    parser.add_argument("--image-a", required=True, help="Path to reference image A")
    parser.add_argument("--image-b", required=True, help="Path to query image B")
    parser.add_argument("--out", default="output", help="Directory to save inlier match visualization")
    return parser.parse_args()


def main():
    args = parse_args()
    path_a = Path(args.image_a)
    path_b = Path(args.image_b)

    result = match_pair(str(path_a), str(path_b))

    if result.status != "SUCCESS":
        print(f"Error: {result.error_message}", file=sys.stderr)
        sys.exit(1)

    print(f"Keypoints A: {result.keypoints_a}")
    print(f"Keypoints B: {result.keypoints_b}")
    print(f"Total matches: {result.total_matches}")
    print(f"Good matches: {result.good_matches}")
    print(f"RANSAC inliers: {result.inliers}")
    print(f"Inlier ratio: {result.inlier_ratio:.4f}")
    print(f"Confidence: {result.confidence:.1f}%")
    print(f"Band: {result.band}")
    print(f"Processing time: {result.time_ms:.1f} ms")

    if args.out:
        out_dir = Path(args.out)
        out_name = f"{path_a.stem}__{path_b.stem}.jpg"
        out_path = out_dir / out_name

        if result.color_a is not None and result.color_b is not None:
            saved = MatchVisualizer.draw_and_save_inliers(
                image_a_bgr=result.color_a,
                keypoints_a=result.keypoints_a_objs,
                image_b_bgr=result.color_b,
                keypoints_b=result.keypoints_b_objs,
                inlier_matches=result.inlier_matches,
                output_path=str(out_path),
            )
            if saved:
                print(f"Saved inlier visualization: {out_path}")


if __name__ == "__main__":
    main()
