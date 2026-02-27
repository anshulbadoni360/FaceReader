"""FastAPI application - Main entry point."""

import time
from pathlib import Path
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.models.schemas import EmotionResponse, HealthResponse
from app.services.analyzer import get_analyzer
from app.utils.constants import EMOTION_CONFIG, AU_LIST, AU_DESCRIPTIONS
from app.job_queue import job_queue
from app.worker import get_worker
from app.database import db


app = FastAPI(
    root_path=settings.ROOT_PATH,
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Fast Face Emotion Analysis API using Action Units",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    print(f"🚀 Starting {settings.APP_NAME} v{settings.VERSION}")
    get_analyzer()
    print("✅ Model loaded")
    
    # Start worker threads for queue processing
    worker = get_worker()
    worker.start()
    print("✅ Worker threads started")


@app.on_event("shutdown")
async def shutdown():
    print("🛑 Shutting down...")
    worker = get_worker()
    worker.stop()
    print("✅ Worker threads stopped")

def validate_file(file: UploadFile) -> None:
    ext = file.filename.split(".")[-1].lower() if file.filename else ""
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Invalid file type. Allowed: {settings.ALLOWED_EXTENSIONS}")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")


async def save_upload_file(file: UploadFile, contents: bytes | None = None) -> str:
    """Save uploaded file and return path."""
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / file.filename
    if contents is None:
        contents = await file.read()
    
    with open(file_path, "wb") as f:
        f.write(contents)
    
    return str(file_path)


@app.get("/", tags=["Root"])
async def root():
    return {"name": settings.APP_NAME, "version": settings.VERSION, "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    analyzer = get_analyzer()
    return {"status": "healthy", "version": settings.VERSION, "model_loaded": analyzer.model is not None}


@app.post("/submit", tags=["Queue"])
async def submit_image(
    file: UploadFile = File(..., description="Image file to analyze")
) -> Dict[str, Any]:
    """
    Submit image to processing queue.
    
    Returns job_id which can be used to retrieve results later.
    """
    validate_file(file)
    
    contents = await file.read()
    if len(contents) > settings.MAX_FILE_SIZE:
        raise HTTPException(400, "File too large")
    
    # Save file
    file_path = await save_upload_file(file, contents)
    
    # Submit to queue
    try:
        job_id = job_queue.submit(file.filename, file_path)
        return {
            "job_id": job_id,
            "filename": file.filename,
            "status": "queued",
            "message": "Image submitted to processing queue"
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/job/{job_id}", tags=["Queue"])
async def get_job_status(job_id: str) -> Dict[str, Any]:
    """Get job status and details."""
    job = db.get_job(job_id)
    
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    
    return job.to_dict()


@app.get("/result/{job_id}", response_model=EmotionResponse, tags=["Queue"])
async def get_result(job_id: str) -> Dict[str, Any]:
    """
    Get analysis result for a job.
    
    Returns the emotion analysis result if job is completed.
    """
    job = db.get_job(job_id)
    
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    
    if job.status.value == "pending":
        raise HTTPException(202, "Job is still pending")
    elif job.status.value == "processing":
        raise HTTPException(202, "Job is still processing")
    elif job.status.value == "failed":
        raise HTTPException(500, f"Job failed: {job.error}")
    
    return JSONResponse(content=job.result)


@app.get("/queue/status", tags=["Queue"])
async def queue_status() -> Dict[str, Any]:
    """Get queue status."""
    return {
        "queue_size": job_queue.size(),
        "max_size": 100,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SYNCHRONOUS ENDPOINTS (Direct Processing)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/analyze", response_model=EmotionResponse, tags=["Analysis"])
async def analyze(
    file: UploadFile = File(..., description="Image file"),
    try_rotations: bool = Query(True, description="Auto-rotate if no face found")
):
    """Analyze facial emotions from uploaded image (direct, synchronous)."""
    validate_file(file)
    
    start = time.perf_counter()
    contents = await file.read()
    
    if len(contents) > settings.MAX_FILE_SIZE:
        raise HTTPException(400, "File too large")
    
    result = get_analyzer().analyze_bytes(contents, try_rotations)
    result["ProcessingTimeMs"] = round((time.perf_counter() - start) * 1000, 2)
    
    return JSONResponse(content=result)


@app.post("/analyze/path", response_model=EmotionResponse, tags=["Analysis"])
async def analyze_path(
    image_path: str = Query(..., description="Path to image"),
    try_rotations: bool = Query(True)
):
    """Analyze from local file path."""
    if not Path(image_path).exists():
        raise HTTPException(404, f"Image not found: {image_path}")
    
    start = time.perf_counter()
    result = get_analyzer().analyze(image_path, try_rotations)
    result["ProcessingTimeMs"] = round((time.perf_counter() - start) * 1000, 2)
    
    return JSONResponse(content=result)


@app.post("/analyze/batch", tags=["Analysis"])
async def analyze_batch(
    files: List[UploadFile] = File(..., description="Multiple images"),
    try_rotations: bool = Query(True)
):
    """Analyze multiple images."""
    if len(files) > 10:
        raise HTTPException(400, "Maximum 10 files per batch")
    
    results = []
    analyzer = get_analyzer()
    
    for file in files:
        try:
            validate_file(file)
            contents = await file.read()
            result = analyzer.analyze_bytes(contents, try_rotations)
            result["filename"] = file.filename
        except Exception as e:
            result = {"filename": file.filename, "FaceAnalyzed": False, "Error": str(e)}
        results.append(result)
    
    return {"results": results, "count": len(results)}


# ══════════════════════════════════════════════════════════════════════════════
# INFO ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/emotions", tags=["Info"])
async def emotions():
    """List detectable emotions."""
    return {"emotions": list(EMOTION_CONFIG.keys()), "config": EMOTION_CONFIG}


@app.get("/action-units", tags=["Info"])
async def action_units():
    """List Action Units."""
    return {"action_units": [{"code": au, "description": AU_DESCRIPTIONS.get(au, "")} for au in AU_LIST]}
