import argparse
import csv
from pathlib import Path
import time
from typing import Dict, List

from src.config import FeatureConfig, PipelineConfig
from src.extractor import FeatureCache
from src.image_matcher import ImageMatcher


def run_ablation_study(pairs_csv_path: str, data_dir_path: str):
    pairs_file = Path(pairs_csv_path)
    data_dir = Path(data_dir_path)

    with open(pairs_file, mode="r", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))

    print("=" * 60)
    print("ABLATION STUDY: SIFT vs RootSIFT (100 Benchmark Pairs)")
    print("=" * 60)

    for method in ["sift", "rootsift"]:
        cfg = PipelineConfig(feature=FeatureConfig(method=method, contrast_threshold=0.01))
        cache = FeatureCache()
        matcher = ImageMatcher(config=cfg, cache=cache)

        same_inliers: List[int] = []
        lookalike_inliers: List[int] = []
        diff_inliers: List[int] = []
        start_time = time.perf_counter()

        for row in pairs:
            pa = data_dir / row["image_a"]
            pb = data_dir / row["image_b"]
            res = matcher.match(str(pa), str(pb))

            if row["label"] == "same":
                same_inliers.append(res.inliers)
            elif row["pair_type"] == "lookalike":
                lookalike_inliers.append(res.inliers)
            else:
                diff_inliers.append(res.inliers)

        elapsed = time.perf_counter() - start_time
        min_same = min(same_inliers)
        max_lookalike = max(lookalike_inliers)
        gap = min_same - max_lookalike

        print(f"\nMethod: {method.upper()}")
        print(f"  Total evaluation time: {elapsed:.2f}s ({elapsed/len(pairs)*1000:.1f} ms/pair)")
        print(f"  Same inliers:       min={min_same:<4} max={max(same_inliers):<4} avg={sum(same_inliers)/len(same_inliers):.1f}")
        print(f"  Lookalike inliers:  min={min(lookalike_inliers):<4} max={max_lookalike:<4} avg={sum(lookalike_inliers)/len(lookalike_inliers):.1f}")
        print(f"  Different inliers:  min={min(diff_inliers):<4} max={max(diff_inliers):<4} avg={sum(diff_inliers)/len(diff_inliers):.1f}")
        print(f"  Safety Separation Gap (min_same - max_lookalike): {gap} inliers")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parameter ablation experiments.")
    parser.add_argument("--pairs", default="Intern_Assignment_Dataset/pairs.csv", help="Path to pairs.csv")
    parser.add_argument("--data", default="Intern_Assignment_Dataset", help="Base dataset directory")
    args = parser.parse_args()
    run_ablation_study(args.pairs, args.data)
