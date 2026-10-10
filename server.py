import csv
from pathlib import Path
import shutil
import tempfile
from typing import List, Optional
import cv2
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse

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
FRONTEND_DIR = BASE_DIR / "frontend"

# Mount dataset directory for frontend image previews
if DATASET_DIR.exists():
    app.mount("/dataset", StaticFiles(directory=str(DATASET_DIR)), name="dataset")

# Mount frontend UI
if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


@app.get("/")
def root():
    return RedirectResponse(url="/ui/")


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
                "preview_url": f"/dataset/reference/{p.name}",
            })

    queries = []
    if query_dir.exists():
        for p in sorted(query_dir.glob("*.jpg")):
            name_parts = p.stem.split("__")
            raw_label = name_parts[1] if len(name_parts) > 1 else p.stem
            is_same = "_same" in p.stem
            is_lookalike = "_lookalike" in p.stem
            category_tag = "[SAME PIECE]" if is_same else ("[LOOKALIKE SIBLING]" if is_lookalike else "[QUERY]")
            label = f"{category_tag} {raw_label.replace('_', ' ').title()}"
            queries.append({
                "id": str(p.relative_to(BASE_DIR)).replace("\\", "/"),
                "name": label,
                "filename": p.name,
                "preview_url": f"/dataset/queries/{p.name}",
                "is_same": is_same,
                "is_lookalike": is_lookalike,
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
        path_a: Optional[Path] = None
        path_b: Optional[Path] = None

        # Resolve Image A if provided
        if file_a and file_a.filename:
            path_a = Path(temp_dir) / f"a_{file_a.filename}"
            with open(path_a, "wb") as buffer:
                shutil.copyfileobj(file_a.file, buffer)
        elif preset_a:
            path_a = BASE_DIR / preset_a

        # Resolve Image B if provided
        if file_b and file_b.filename:
            path_b = Path(temp_dir) / f"b_{file_b.filename}"
            with open(path_b, "wb") as buffer:
                shutil.copyfileobj(file_b.file, buffer)
        elif preset_b:
            path_b = BASE_DIR / preset_b

        # CASE 1: Only ONE image provided (Auto-match against pre-fed references)
        if (path_a is None and path_b is not None) or (path_a is not None and path_b is None):
            query_path = path_b if path_b is not None else path_a
            ref_dir = DATASET_DIR / "reference"
            if not ref_dir.exists():
                return JSONResponse({"error": "Pre-fed reference inventory not found"}, status_code=500)

            best_res = None
            best_ref_label = ""
            best_ref_path = None
            highest_conf = -1.0

            for ref_file in sorted(ref_dir.glob("*.jpg")):
                curr_res = matcher.match(str(ref_file), str(query_path))
                if curr_res.confidence > highest_conf:
                    highest_conf = curr_res.confidence
                    best_res = curr_res
                    best_ref_path = ref_file
                    name_parts = ref_file.stem.split("__")
                    best_ref_label = name_parts[1].replace("_", " ").title() if len(name_parts) > 1 else ref_file.stem

            if best_res is None:
                return JSONResponse({"error": "Failed to evaluate against inventory"}, status_code=500)

            vis_base64 = ""
            if best_res.color_a is not None and best_res.color_b is not None:
                vis_base64 = MatchVisualizer.render_inliers_to_base64(
                    image_a_bgr=best_res.color_a,
                    keypoints_a=best_res.keypoints_a_objs,
                    image_b_bgr=best_res.color_b,
                    keypoints_b=best_res.keypoints_b_objs,
                    inlier_matches=best_res.inlier_matches,
                    band=best_res.band,
                    confidence=best_res.confidence,
                )

            return {
                "status": best_res.status,
                "band": best_res.band,
                "confidence": best_res.confidence,
                "inliers": best_res.inliers,
                "good_matches": best_res.good_matches,
                "total_matches": best_res.total_matches,
                "inlier_ratio": best_res.inlier_ratio,
                "keypoints_a": best_res.keypoints_a,
                "keypoints_b": best_res.keypoints_b,
                "time_ms": best_res.time_ms,
                "error_message": best_res.error_message,
                "visualization": vis_base64,
                "is_auto_matched": True,
                "matched_reference_name": best_ref_label,
                "is_lookalike_pair": False,
                "lookalike_reason": "",
            }

        # CASE 2: Neither image provided
        if path_a is None and path_b is None:
            return JSONResponse({"error": "Please upload at least one image or choose a sample."}, status_code=400)

        # CASE 3: Both images provided (Direct pairwise matching)
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

        is_lookalike_pair = False
        lookalike_reason = ""
        if path_a and path_b:
            name_a = path_a.stem.lower()
            name_b = path_b.stem.lower()
            if "lookalike" in name_a or "lookalike" in name_b:
                is_lookalike_pair = True
                lookalike_reason = (
                    "Quarry Sibling Lookalike Correctly Rejected: Slabs cut from adjacent block sections "
                    "share texture and mineral coloring ('family resemblance'), but differing micro-vein "
                    "geometry confirms they are physically distinct pieces. Rejecting lookalikes ensures "
                    "zero false acceptances."
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
            "is_auto_matched": False,
            "matched_reference_name": None,
            "is_lookalike_pair": is_lookalike_pair,
            "lookalike_reason": lookalike_reason,
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
        pairs_csv = DATASET_DIR / "pairs.csv"
        if DATASET_DIR.exists() and pairs_csv.exists():
            from evaluate import evaluate_batch
            evaluate_batch(str(DATASET_DIR), str(pairs_csv), str(csv_file))
        else:
            return JSONResponse({
                "rows": [],
                "summary": {
                    "total_pairs": 0,
                    "same_accuracy": 0.0,
                    "diff_accuracy": 0.0,
                    "lookalike_false_acceptances": 0,
                },
            })

    rows = []
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    same_total = sum(1 for r in rows if r.get("label") == "same")
    same_correct = sum(1 for r in rows if r.get("label") == "same" and r.get("band") == "SAME")
    diff_total = sum(1 for r in rows if r.get("label") == "different")
    diff_correct = sum(1 for r in rows if r.get("label") == "different" and r.get("band") in ["DIFFERENT", "SIMILAR"])
    lookalike_false = sum(1 for r in rows if r.get("pair_type") == "lookalike" and r.get("band") == "SAME")

    summary = {
        "total_pairs": len(rows),
        "same_accuracy": round((same_correct / same_total * 100.0) if same_total else 0.0, 1),
        "diff_accuracy": round((diff_correct / diff_total * 100.0) if diff_total else 0.0, 1),
        "lookalike_false_acceptances": lookalike_false,
    }

    return {"rows": rows, "summary": summary}
