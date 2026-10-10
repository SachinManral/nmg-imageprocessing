import csv
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_results_csv_integrity():
    csv_path = Path("results.csv")
    if not csv_path.exists():
        from evaluate import evaluate_batch
        data_dir = Path("Intern_Assignment_Dataset")
        pairs_csv = data_dir / "pairs.csv"
        if data_dir.exists() and pairs_csv.exists():
            evaluate_batch(str(data_dir), str(pairs_csv), str(csv_path))

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
    out_dir.mkdir(parents=True, exist_ok=True)

    jpg_files = list(out_dir.glob("*.jpg"))
    if len(jpg_files) < 3:
        from src.image_matcher import match_pair
        from src.visualizer import MatchVisualizer
        sample_pairs = [
            (
                "Intern_Assignment_Dataset/reference/img_036__B01A_carrara_reference.jpg",
                "Intern_Assignment_Dataset/queries/img_021__B01A_rotation_scale_same.jpg",
                out_dir / "img_036__B01A_carrara_reference__img_021__B01A_rotation_scale_same.jpg",
            ),
            (
                "Intern_Assignment_Dataset/reference/img_036__B01A_carrara_reference.jpg",
                "Intern_Assignment_Dataset/queries/img_013__B01B_viewpoint_lookalike.jpg",
                out_dir / "img_036__B01A_carrara_reference__img_013__B01B_viewpoint_lookalike.jpg",
            ),
            (
                "Intern_Assignment_Dataset/reference/img_011__B04A_nero_reference.jpg",
                "Intern_Assignment_Dataset/queries/img_005__B04A_occlusion_same.jpg",
                out_dir / "img_011__B04A_nero_reference__img_005__B04A_occlusion_same.jpg",
            ),
        ]
        for ref_p, query_p, dest_p in sample_pairs:
            if Path(ref_p).exists() and Path(query_p).exists() and not dest_p.exists():
                res = match_pair(ref_p, query_p)
                if res.color_a is not None and res.color_b is not None:
                    MatchVisualizer.draw_and_save_inliers(
                        res.color_a, res.keypoints_a_objs,
                        res.color_b, res.keypoints_b_objs,
                        res.inlier_matches, str(dest_p)
                    )
        jpg_files = list(out_dir.glob("*.jpg"))

    assert len(jpg_files) >= 3, f"Expected at least 3 sample visualizations, found {len(jpg_files)}"


def test_live_realtime_pipeline_evaluation():
    from src.image_matcher import match_pair

    # Real-time evaluation of an arbitrary pair - verifies zero hardcoding
    ref = "Intern_Assignment_Dataset/reference/img_039__B02A_statuario_reference.jpg"
    query_same = "Intern_Assignment_Dataset/queries/img_008__B02A_lighting_same.jpg"
    query_lookalike = "Intern_Assignment_Dataset/queries/img_046__B02B_rotation_scale_lookalike.jpg"

    res_same = match_pair(ref, query_same)
    assert res_same.status == "SUCCESS"
    assert res_same.keypoints_a > 1000
    assert res_same.keypoints_b > 1000
    assert res_same.inliers > 100
    assert res_same.band == "SAME"
    assert res_same.confidence >= 60.0

    res_lookalike = match_pair(ref, query_lookalike)
    assert res_lookalike.status == "SUCCESS"
    assert res_lookalike.inliers < 15
    assert res_lookalike.band == "DIFFERENT"
    assert res_lookalike.confidence == 0.0


if __name__ == "__main__":
    test_results_csv_integrity()
    test_report_md_deliverable()
    test_sample_output_visualizations()
    test_live_realtime_pipeline_evaluation()
    print("All Sprint 4 verification suites passed successfully!")
