import csv
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_results_csv_integrity():
    csv_path = Path("results.csv")
    assert csv_path.exists(), "results.csv deliverable not found"

    with open(csv_path, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 100, f"Expected 100 evaluated pairs, got {len(rows)}"

    required_fields = [
        "image_a", "image_b", "label", "pair_type",
        "keypoints_a", "keypoints_b", "total_matches",
        "good_matches", "inliers", "inlier_ratio",
        "confidence", "band", "time_ms"
    ]
    for field in required_fields:
        assert field in rows[0], f"Missing required column: {field}"

    same_rows = [r for r in rows if r["label"] == "same"]
    lookalike_rows = [r for r in rows if r["pair_type"] == "lookalike"]
    diff_rows = [r for r in rows if r["pair_type"] == "different"]

    assert len(same_rows) == 10
    assert len(lookalike_rows) == 10
    assert len(diff_rows) == 80

    # Verification: 100% same pairs classified SAME
    same_correct = sum(1 for r in same_rows if r["band"] == "SAME")
    assert same_correct == 10, f"Expected 10 SAME classifications, got {same_correct}"

    # Verification: 0 lookalike false acceptances
    lookalike_false_same = sum(1 for r in lookalike_rows if r["band"] == "SAME")
    assert lookalike_false_same == 0, f"Found {lookalike_false_same} lookalike false acceptances"

    # Verification: all different pairs rejected
    diff_rejected = sum(1 for r in diff_rows if r["band"] in ["DIFFERENT", "SIMILAR"])
    assert diff_rejected == 80


def test_report_md_deliverable():
    report_path = Path("REPORT.md")
    assert report_path.exists(), "REPORT.md deliverable not found"

    content = report_path.read_text(encoding="utf-8")
    assert "Algorithm Selection" in content
    assert "Confidence Scoring" in content
    assert "Benchmark Results" in content
    assert "Challenges Faced" in content
    assert "Potential Improvements" in content


def test_sample_output_visualizations():
    out_dir = Path("output")
    assert out_dir.exists(), "output/ directory not found"

    jpg_files = list(out_dir.glob("*.jpg"))
    assert len(jpg_files) >= 3, f"Expected at least 3 sample visualizations, found {len(jpg_files)}"


if __name__ == "__main__":
    test_results_csv_integrity()
    test_report_md_deliverable()
    test_sample_output_visualizations()
    print("All Sprint 4 verification suites passed successfully!")
