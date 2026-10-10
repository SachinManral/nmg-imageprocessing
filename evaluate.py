import argparse
import csv
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

from src.config import PipelineConfig
from src.extractor import FeatureCache
from src.image_matcher import ImageMatcher


def parse_args():
    parser = argparse.ArgumentParser(description="Batch evaluation of image pairs.")
    parser.add_argument("--data", required=True, help="Path to base data directory")
    parser.add_argument("--pairs", required=True, help="Path to pairs.csv file")
    parser.add_argument("--out", default="results.csv", help="Path to output results.csv")
    return parser.parse_args()


def resolve_image_path(data_dir: Path, image_rel_path: str) -> Path:
    # Check direct relative path
    direct = data_dir / image_rel_path
    if direct.exists():
        return direct

    # Check common subdirectories
    for subdir in ["images", "reference", "queries"]:
        candidate = data_dir / subdir / image_rel_path
        if candidate.exists():
            return candidate
        # If relative path already has the filename only
        candidate_flat = data_dir / subdir / Path(image_rel_path).name
        if candidate_flat.exists():
            return candidate_flat

    # Check if absolute path
    abs_path = Path(image_rel_path)
    if abs_path.exists():
        return abs_path

    # Return direct as fallback for error logging
    return direct


def evaluate_batch(data_dir_str: str, pairs_csv_str: str, out_csv_str: str):
    data_dir = Path(data_dir_str)
    pairs_csv = Path(pairs_csv_str)
    out_csv = Path(out_csv_str)

    if not pairs_csv.exists():
        print(f"Error: pairs file not found at {pairs_csv}", file=sys.stderr)
        sys.exit(1)

    # Initialize matcher with feature cache
    cache = FeatureCache()
    matcher = ImageMatcher(config=PipelineConfig(), cache=cache)

    records = []
    with open(pairs_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    total_pairs = len(records)
    print(f"Evaluating {total_pairs} pairs from {pairs_csv}...")

    results_rows = []
    has_labels = True

    for idx, row in enumerate(records, 1):
        img_a_rel = row.get("image_a", "").strip()
        img_b_rel = row.get("image_b", "").strip()
        label = row.get("label", "unknown").strip().lower()
        pair_type = row.get("pair_type", "unknown").strip().lower()

        if label in ["unknown", ""]:
            has_labels = False

        path_a = resolve_image_path(data_dir, img_a_rel)
        path_b = resolve_image_path(data_dir, img_b_rel)

        match_res = matcher.match(str(path_a), str(path_b))

        results_rows.append({
            "image_a": img_a_rel,
            "image_b": img_b_rel,
            "label": label,
            "pair_type": pair_type,
            "keypoints_a": match_res.keypoints_a,
            "keypoints_b": match_res.keypoints_b,
            "total_matches": match_res.total_matches,
            "good_matches": match_res.good_matches,
            "inliers": match_res.inliers,
            "inlier_ratio": f"{match_res.inlier_ratio:.4f}",
            "confidence": f"{match_res.confidence:.1f}",
            "band": match_res.band,
            "time_ms": f"{match_res.time_ms:.1f}",
        })

    # Write results.csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_a",
        "image_b",
        "label",
        "pair_type",
        "keypoints_a",
        "keypoints_b",
        "total_matches",
        "good_matches",
        "inliers",
        "inlier_ratio",
        "confidence",
        "band",
        "time_ms",
    ]

    with open(out_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)

    print(f"Wrote {len(results_rows)} results to {out_csv}")

    # If hidden set (labels unknown), skip printing metric table
    if not has_labels:
        print("Labels are unknown (hidden test set). Evaluation complete without metrics calculation.")
        return

    # Print accuracy summary
    print_accuracy_report(results_rows)


def print_accuracy_report(results_rows: List[Dict]):
    same_total = 0
    same_correct = 0

    diff_total = 0
    diff_correct = 0

    lookalike_total = 0
    lookalike_false_same = 0

    pair_type_stats: Dict[str, Dict[str, int]] = {}

    for row in results_rows:
        label = row["label"].lower()
        ptype = row["pair_type"].lower()
        band = row["band"].upper()

        if ptype not in pair_type_stats:
            pair_type_stats[ptype] = {"SAME": 0, "SIMILAR": 0, "DIFFERENT": 0, "total": 0}
        pair_type_stats[ptype][band] += 1
        pair_type_stats[ptype]["total"] += 1

        if label == "same":
            same_total += 1
            if band == "SAME":
                same_correct += 1
        elif label == "different":
            diff_total += 1
            if band in ["DIFFERENT", "SIMILAR"]:
                diff_correct += 1
            if ptype == "lookalike":
                lookalike_total += 1
                if band == "SAME":
                    lookalike_false_same += 1

    print("\n" + "=" * 50)
    print("ACCURACY SUMMARY")
    print("=" * 50)

    if same_total > 0:
        same_pct = (same_correct / float(same_total)) * 100.0
        print(f"Same pairs classified SAME: {same_correct}/{same_total} ({same_pct:.1f}%)")

    if diff_total > 0:
        diff_pct = (diff_correct / float(diff_total)) * 100.0
        print(f"Different pairs correctly rejected: {diff_correct}/{diff_total} ({diff_pct:.1f}%)")

    if lookalike_total > 0:
        print(f"Lookalike pairs wrongly called SAME: {lookalike_false_same}/{lookalike_total}")

    print("\nRESULTS BROKEN DOWN BY PAIR TYPE:")
    header = f"{'Pair Type':<20} | {'Total':<6} | {'SAME':<6} | {'SIMILAR':<8} | {'DIFFERENT':<10}"
    print(header)
    print("-" * len(header))
    for ptype, counts in sorted(pair_type_stats.items()):
        print(
            f"{ptype:<20} | {counts['total']:<6} | {counts['SAME']:<6} | "
            f"{counts['SIMILAR']:<8} | {counts['DIFFERENT']:<10}"
        )
    print("=" * 50 + "\n")


if __name__ == "__main__":
    cli_args = parse_args()
    evaluate_batch(cli_args.data, cli_args.pairs, cli_args.out)
