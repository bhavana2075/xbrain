"""
api/main.py
FastAPI backend for X-Brain.
TensorFlow and PyTorch are imported lazily (on first /analyze request)
so uvicorn can bind the port immediately without timing out.
"""

import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# ── Lightweight imports only (no TF/Torch here) ───────────────────────────────
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("xbrain")

# ── Config ────────────────────────────────────────────────────────────────────
CLF_WEIGHTS = ROOT / os.getenv("CLF_WEIGHTS", "checkpoints/EfficientNetB0_BrainTumor_full.weights.h5")
SEG_WEIGHTS = ROOT / os.getenv("SEG_WEIGHTS", "checkpoints/SwinUNETR_Segmentation_best.pth")
FAISS_INDEX_PATH = ROOT / os.getenv("FAISS_INDEX_PATH", "checkpoints/faiss_index.index")
LOAD_MODELS_ON_STARTUP = os.getenv("LOAD_MODELS_ON_STARTUP", "true").lower() == "true"

# ── Global holders ────────────────────────────────────────────────────────────
MODELS = {}
MODS = {}   # lazy-imported heavy modules


def _import_heavy():
    """Import TensorFlow, PyTorch, OpenCV etc. only when first needed."""
    if MODS:
        return
    log.info("Importing heavy modules (TF + Torch)...")
    import numpy as np
    import cv2
    from models.classifier import build_classifier, classify, get_gradcam_overlay
    from models.segmentor import build_segmentor, segment, compute_tumor_stats, get_segmentation_overlay
    from utils.image_utils import read_image_from_bytes, ndarray_to_base64, mask_to_base64
    from utils.clinical_knowledge import generate_clinical_report
    from utils.rag_pipeline import (
        generate_rag_report, answer_question, translate_text,
        get_supported_languages, build_index,
    )
    MODS.update(locals())
    log.info("Heavy modules imported")


def _load_models():
    _import_heavy()
    if "classifier" not in MODELS:
        if CLF_WEIGHTS.exists():
            MODELS["classifier"] = MODS["build_classifier"](str(CLF_WEIGHTS))
            log.info("Classifier loaded")
        else:
            log.error(f"CLF weights not found: {CLF_WEIGHTS}")
    if "segmentor" not in MODELS:
        if SEG_WEIGHTS.exists():
            seg, device = MODS["build_segmentor"](str(SEG_WEIGHTS))
            MODELS["segmentor"] = seg
            MODELS["device"] = device
            log.info("Segmentor loaded")
        else:
            log.error(f"SEG weights not found: {SEG_WEIGHTS}")


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    if LOAD_MODELS_ON_STARTUP:
        log.info("Loading models at startup...")
        _load_models()
        try:
            _import_heavy()
            MODS["build_index"](force=False)
        except Exception as e:
            log.warning(f"Index build skipped: {e}")
    else:
        log.info("Lazy mode - models load on first /analyze request")
    log.info("X-Brain API ready")
    yield
    MODELS.clear()
    log.info("Models unloaded")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="X-Brain API", description="Explainable AI Brain Tumor Analysis", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Schemas ───────────────────────────────────────────────────────────────────
class ClassificationResult(BaseModel):
    class_name: str
    class_idx: int
    confidence: float
    probabilities: dict
    has_tumor: bool

class SegmentationResult(BaseModel):
    tumor_area_pct: float
    tumor_pixels: int
    total_pixels: int
    has_mask: bool
    skipped: bool

class RAGResult(BaseModel):
    llm_report: str
    source: str
    retrieved_docs: list
    language: str

class AnalysisResponse(BaseModel):
    classification: ClassificationResult
    segmentation: SegmentationResult
    clinical_report: dict
    rag_report: RAGResult
    images: dict
    inference_time_ms: float

class QARequest(BaseModel):
    question: str
    report_context: str
    class_name: str
    conversation_history: Optional[list] = []
    language: str = "en"

class TranslateRequest(BaseModel):
    text: str
    language: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "X-Brain API v2.0",
        "status": "running",
        "models": {
            "classifier": "loaded" if "classifier" in MODELS else "missing",
            "segmentor": "loaded" if "segmentor" in MODELS else "missing",
        },
        "rag": "ready" if FAISS_INDEX_PATH.exists() else "not_built",
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if ("classifier" in MODELS and "segmentor" in MODELS) else "degraded",
        "classifier": "classifier" in MODELS,
        "segmentor": "segmentor" in MODELS,
        "groq_llm": bool(os.getenv("GROQ_API_KEY")),
        "rag_index": FAISS_INDEX_PATH.exists(),
    }


@app.get("/languages")
def languages():
    _import_heavy()
    return MODS["get_supported_languages"]()


@app.post("/index/build")
def rebuild_index(force: bool = False):
    _import_heavy()
    success = MODS["build_index"](force=force)
    return {"success": success, "message": "Index built" if success else "Build failed or no PDFs found"}


@app.post("/qa")
def question_answer(req: QARequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    _import_heavy()
    answer = MODS["answer_question"](
        question=req.question,
        report_context=req.report_context,
        class_name=req.class_name,
        conversation_history=req.conversation_history,
        language=req.language,
    )
    return {"answer": answer, "language": req.language}


@app.post("/translate")
def translate(req: TranslateRequest):
    _import_heavy()
    translated = MODS["translate_text"](req.text, req.language)
    return {"translated": translated, "language": req.language}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...), language: str = "en"):
    # Lazy load on first request
    if "classifier" not in MODELS or "segmentor" not in MODELS:
        log.info("Lazy loading models on first request...")
        _load_models()

    if "classifier" not in MODELS:
        raise HTTPException(status_code=503, detail="Classifier model not loaded.")
    if "segmentor" not in MODELS:
        raise HTTPException(status_code=503, detail="Segmentor model not loaded.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        img_rgb = MODS["read_image_from_bytes"](raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    np = MODS["np"]
    cv2 = MODS["cv2"]
    t0 = time.time()

    clf_result = MODS["classify"](MODELS["classifier"], img_rgb)
    log.info(f"Classification: {clf_result['class_name']} ({clf_result['confidence']:.2%})")

    heatmap_rgb, gradcam_overlay = MODS["get_gradcam_overlay"](
        MODELS["classifier"], img_rgb, pred_index=clf_result["class_idx"]
    )

    skipped = not clf_result["has_tumor"]
    if skipped:
        mask = np.zeros((224, 224), dtype=np.float32)
        seg_overlay = np.zeros((224, 224, 3), dtype=np.uint8)
    else:
        mask = MODS["segment"](MODELS["segmentor"], img_rgb, MODELS["device"])
        seg_overlay = MODS["get_segmentation_overlay"](img_rgb, mask)

    tumor_stats = MODS["compute_tumor_stats"](mask)

    report = MODS["generate_clinical_report"](
        class_name=clf_result["class_name"],
        confidence=clf_result["confidence"],
        probabilities=clf_result["probabilities"],
        tumor_area_pct=tumor_stats["tumor_area_pct"],
        has_mask=tumor_stats["has_mask"],
        language=language,
    )

    rag_result = MODS["generate_rag_report"](
        class_name=clf_result["class_name"],
        confidence=clf_result["confidence"],
        tumor_area_pct=tumor_stats["tumor_area_pct"],
        has_mask=tumor_stats["has_mask"],
        probabilities=clf_result["probabilities"],
        language=language,
    )

    inference_time = round((time.time() - t0) * 1000, 1)
    log.info(f"Total inference time: {inference_time} ms")

    img_224 = cv2.resize(img_rgb, (224, 224))
    images = {
        "original":        MODS["ndarray_to_base64"](img_224),
        "gradcam_heatmap": MODS["ndarray_to_base64"](heatmap_rgb),
        "gradcam_overlay": MODS["ndarray_to_base64"](gradcam_overlay),
        "seg_mask":        MODS["mask_to_base64"](mask),
        "seg_overlay":     MODS["ndarray_to_base64"](seg_overlay),
    }

    return AnalysisResponse(
        classification=ClassificationResult(**clf_result),
        segmentation=SegmentationResult(
            tumor_area_pct=tumor_stats["tumor_area_pct"],
            tumor_pixels=tumor_stats["tumor_pixels"],
            total_pixels=tumor_stats["total_pixels"],
            has_mask=tumor_stats["has_mask"],
            skipped=skipped,
        ),
        clinical_report=report,
        rag_report=RAGResult(**rag_result),
        images=images,
        inference_time_ms=inference_time,
    )
