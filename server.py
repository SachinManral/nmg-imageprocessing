import csv
from pathlib import Path
import shutil
import tempfile
from typing import List, Optional
import cv2
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import PipelineConfig
from src.extractor import FeatureCache
from src.image_matcher import ImageMatcher
from src.visualizer import MatchVisualizer


app = FastAPI(title="Image Matching & Verification REST API")

# Enable CORS for decoupled frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Intern_Assignment_Dataset"

cache = FeatureCache()
matcher = ImageMatcher(config=PipelineConfig(), cache=cache)


@app.get("/api/health")
def healthcheck():
    return {"status": "ok", "service": "image-matching-api", "version": "1.0.0"}


@app.get("/api/presets")
def get_presets():
    ref_dir = DATASET_DIR / "reference"
    query_dir = DATASET_DIR / "queries"

    references = []
    if ref_dir.exists():
        for p in sorted(ref_dir.glob("*.jpg")):
            name_parts = p.stem.split("__")
            label = name_parts[1].replace("_", " ").title() if len(name_parts) > 1 else p.stem
            references.append({
                "id": str(p.relative_to(BASE_DIR)).replace("\\", "/"),
                "name": label,
                "filename": p.name,
            })

    queries = []
    if query_dir.exists():
        for p in sorted(query_dir.glob("*.jpg")):
            name_parts = p.stem.split("__")
            label = name_parts[1].replace("_", " ").title() if len(name_parts) > 1 else p.stem
            queries.append({
                "id": str(p.relative_to(BASE_DIR)).replace("\\", "/"),
                "name": label,
                "filename": p.name,
            })

    return {"references": references, "queries": queries}


@app.post("/api/match")
async def match_endpoint(
    file_a: Optional[UploadFile] = File(None),
    file_b: Optional[UploadFile] = File(None),
    preset_a: Optional[str] = Form(None),
    preset_b: Optional[str] = Form(None),
):
    temp_dir = tempfile.mkdtemp()
    try:
        # Resolve Image A
        if file_a and file_a.filename:
            path_a = Path(temp_dir) / file_a.filename
            with open(path_a, "wb") as buffer:
                shutil.copyfileobj(file_a.file, buffer)
        elif preset_a:
            path_a = BASE_DIR / preset_a
        else:
            return JSONResponse({"error": "Image A is required"}, status_code=400)

        # Resolve Image B
        if file_b and file_b.filename:
            path_b = Path(temp_dir) / file_b.filename
            with open(path_b, "wb") as buffer:
                shutil.copyfileobj(file_b.file, buffer)
        elif preset_b:
            path_b = BASE_DIR / preset_b
        else:
            return JSONResponse({"error": "Image B is required"}, status_code=400)

        result = matcher.match(str(path_a), str(path_b))

        vis_base64 = ""
        if result.color_a is not None and result.color_b is not None:
            vis_base64 = MatchVisualizer.render_inliers_to_base64(
                image_a_bgr=result.color_a,
                keypoints_a=result.keypoints_a_objs,
                image_b_bgr=result.color_b,
                keypoints_b=result.keypoints_b_objs,
                inlier_matches=result.inlier_matches,
                band=result.band,
                confidence=result.confidence,
            )

        return {
            "status": result.status,
            "band": result.band,
            "confidence": result.confidence,
            "inliers": result.inliers,
            "good_matches": result.good_matches,
            "total_matches": result.total_matches,
            "inlier_ratio": result.inlier_ratio,
            "keypoints_a": result.keypoints_a,
            "keypoints_b": result.keypoints_b,
            "time_ms": result.time_ms,
            "error_message": result.error_message,
            "visualization": vis_base64,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/api/gallery-search")
async def gallery_search_endpoint(
    query_file: Optional[UploadFile] = File(None),
    preset_query: Optional[str] = Form(None),
):
    temp_dir = tempfile.mkdtemp()
    try:
        if query_file and query_file.filename:
            query_path = Path(temp_dir) / query_file.filename
            with open(query_path, "wb") as buffer:
                shutil.copyfileobj(query_file.file, buffer)
        elif preset_query:
            query_path = BASE_DIR / preset_query
        else:
            return JSONResponse({"error": "Query image is required"}, status_code=400)

        ref_dir = DATASET_DIR / "reference"
        if not ref_dir.exists():
            return JSONResponse({"error": "Reference directory not found"}, status_code=500)

        candidates = []
        for ref_img in sorted(ref_dir.glob("*.jpg")):
            res = matcher.match(str(ref_img), str(query_path))
            name_parts = ref_img.stem.split("__")
            label = name_parts[1].replace("_", " ").title() if len(name_parts) > 1 else ref_img.stem

            candidates.append({
                "reference_name": label,
                "filename": ref_img.name,
                "band": res.band,
                "confidence": res.confidence,
                "inliers": res.inliers,
                "inlier_ratio": res.inlier_ratio,
                "time_ms": res.time_ms,
            })

        # Sort by confidence descending
        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        top_match = candidates[0] if candidates and candidates[0]["band"] == "SAME" else None

        return {
            "top_match": top_match,
            "candidates": candidates,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/api/benchmark-results")
def get_benchmark_results():
    csv_file = BASE_DIR / "results.csv"
    if not csv_file.exists():
        return JSONResponse({"rows": [], "summary": {}})

    rows = []
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    same_total = sum(1 for r in rows if r["label"] == "same")
    same_correct = sum(1 for r in rows if r["label"] == "same" and r["band"] == "SAME")
    diff_total = sum(1 for r in rows if r["label"] == "different")
    diff_correct = sum(1 for r in rows if r["label"] == "different" and r["band"] in ["DIFFERENT", "SIMILAR"])
    lookalike_false = sum(1 for r in rows if r["pair_type"] == "lookalike" and r["band"] == "SAME")

    summary = {
        "total_pairs": len(rows),
        "same_accuracy": round((same_correct / same_total * 100.0) if same_total else 0, 1),
        "diff_accuracy": round((diff_correct / diff_total * 100.0) if diff_total else 0, 1),
        "lookalike_false_acceptances": lookalike_false,
    }

    return {"rows": rows, "summary": summary}
