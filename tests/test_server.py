import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from starlette.testclient import TestClient
from server import app


client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"


def test_presets_endpoint():
    res = client.get("/api/presets")
    assert res.status_code == 200
    data = res.json()
    assert "references" in data
    assert "queries" in data
    assert len(data["references"]) == 5
    assert len(data["queries"]) == 20


def test_benchmark_results_endpoint():
    res = client.get("/api/benchmark-results")
    assert res.status_code == 200
    data = res.json()
    assert "rows" in data
    assert "summary" in data
    assert data["summary"]["same_accuracy"] == 100.0
    assert data["summary"]["lookalike_false_acceptances"] == 0


def test_match_endpoint_via_presets():
    res = client.post("/api/match", data={
        "preset_a": "Intern_Assignment_Dataset/reference/img_036__B01A_carrara_reference.jpg",
        "preset_b": "Intern_Assignment_Dataset/queries/img_021__B01A_rotation_scale_same.jpg",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["band"] == "SAME"
    assert data["confidence"] >= 60.0
    assert data["inliers"] > 100
    assert data["visualization"].startswith("data:image/jpeg;base64,")


if __name__ == "__main__":
    test_health_endpoint()
    test_presets_endpoint()
    test_benchmark_results_endpoint()
    test_match_endpoint_via_presets()
    print("All server REST API tests passed successfully!")
