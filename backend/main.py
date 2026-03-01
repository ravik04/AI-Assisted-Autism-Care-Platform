"""
Autism AI Screening — FastAPI Backend (v4 Multi-Modal)
=====================================================
REST API serving 5 trained models + 4 cooperative agents.

Models:
  1. Face Classifier      (MobileNetV2, .keras)
  2. Behavior LSTM        (MobileNetV2+LSTM, .keras)
  3. Questionnaire XGB    (XGBoost, .pkl)
  4. Eye-Tracking XGB     (XGBoost, .pkl)
  5. Pose/Skeleton XGB    (XGBoost, .pkl)
  6. CARS Severity Ridge  (Ridge regression, .pkl) — enrichment

Endpoints:
  POST /api/analyze          — image/video → face + behavior scores + agents
  POST /api/questionnaire    — screening questionnaire → risk score + agents
  POST /api/fuse             — combine all modality scores → full pipeline
  GET  /api/history          — score history
  POST /api/clear            — reset history
  GET  /api/status           — model status
  GET  /api/model-info       — metadata for all models
"""

import os, sys, io, json, base64, tempfile
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List

# ── Project imports ────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from agents.screening_agent import screening_agent
from agents.clinical_agent import clinical_agent
from agents.therapy_agent import therapy_agent
from agents.monitoring_agent import monitoring_agent
from utils.gradcam import make_gradcam_heatmap, overlay_gradcam
from utils.logger import save_result
from utils.llm_client import is_llm_available

# ── Paths ──────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(PROJECT_DIR, "saved_models")
IMG_SIZE = (224, 224)

# ── FastAPI App ────────────────────────────────────────────────────────
app = FastAPI(
    title="Autism AI Multi-Modal Screening API",
    description="5-model, 4-agent developmental screening pipeline",
    version="4.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model references ───────────────────────────────────────────
face_model = None
behavior_model = None
feature_extractor = None
gap_layer = None
questionnaire_model = None
questionnaire_scaler = None
questionnaire_meta = None
eye_tracking_model = None
eye_tracking_scaler = None
eye_tracking_meta = None
pose_model = None
pose_scaler = None
pose_meta = None
cars_model = None
cars_scaler = None
cars_meta = None


def _load_pkl(name):
    import joblib
    path = os.path.join(MODELS_DIR, name)
    if os.path.exists(path):
        return joblib.load(path)
    return None


def _load_json(name):
    path = os.path.join(MODELS_DIR, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@app.on_event("startup")
def load_models():
    global face_model, behavior_model, feature_extractor, gap_layer
    global questionnaire_model, questionnaire_scaler, questionnaire_meta
    global eye_tracking_model, eye_tracking_scaler, eye_tracking_meta
    global pose_model, pose_scaler, pose_meta
    global cars_model, cars_scaler, cars_meta

    print("\n╔══════════════════════════════════════════════╗")
    print("║  Loading Autism AI Multi-Modal Models...     ║")
    print("╚══════════════════════════════════════════════╝\n")

    # 1) Face Classifier
    fp = os.path.join(MODELS_DIR, "face_classifier.keras")
    if os.path.exists(fp):
        face_model = tf.keras.models.load_model(fp)
        print("  ✓ Face classifier loaded")
    else:
        print("  ✗ Face classifier not found")

    # 2) Behavior LSTM
    bp = os.path.join(MODELS_DIR, "behavior_lstm.keras")
    if os.path.exists(bp):
        behavior_model = tf.keras.models.load_model(bp)
        print("  ✓ Behavior LSTM loaded")
    else:
        print("  ✗ Behavior LSTM not found")

    # MobileNetV2 feature extractor for behavior
    feature_extractor = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    gap_layer = tf.keras.layers.GlobalAveragePooling2D()
    print("  ✓ MobileNetV2 feature extractor ready")

    # 3) Questionnaire XGBoost
    questionnaire_model = _load_pkl("questionnaire_xgb.pkl")
    questionnaire_scaler = _load_pkl("questionnaire_scaler.pkl")
    questionnaire_meta = _load_json("questionnaire_metadata.json")
    print(f"  {'✓' if questionnaire_model else '✗'} Questionnaire XGBoost")

    # 4) Eye-Tracking XGBoost
    eye_tracking_model = _load_pkl("eye_tracking_xgb.pkl")
    eye_tracking_scaler = _load_pkl("eye_tracking_scaler.pkl")
    eye_tracking_meta = _load_json("eye_tracking_metadata.json")
    print(f"  {'✓' if eye_tracking_model else '✗'} Eye-Tracking XGBoost")

    # 5) Pose/Skeleton XGBoost
    pose_model = _load_pkl("pose_skeleton_xgb.pkl")
    pose_scaler = _load_pkl("pose_skeleton_scaler.pkl")
    pose_meta = _load_json("pose_skeleton_metadata.json")
    print(f"  {'✓' if pose_model else '✗'} Pose/Skeleton XGBoost")

    # 6) CARS Severity Ridge (enrichment)
    cars_model = _load_pkl("eye_tracking_cars_model.pkl")
    cars_scaler = _load_pkl("eye_tracking_cars_scaler.pkl")
    cars_meta = _load_json("eye_tracking_cars_metadata.json")
    print(f"  {'✓' if cars_model else '✗'} CARS Severity Ridge")

    print("\n  All models loaded.\n")


# ── In-memory state ───────────────────────────────────────────────────
score_history: list = []
modality_history: list = []


# ══════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def numpy_to_base64_png(arr: np.ndarray) -> str:
    img = Image.fromarray(np.uint8(arr))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _run_agent_pipeline(fused_score: float, modality_scores: dict, domain_profile: dict = None):
    """Run all 4 agents with full modality context."""
    s_out = screening_agent(fused_score, modality_scores=modality_scores)
    c_out = clinical_agent(s_out, modality_scores=modality_scores, domain_profile=domain_profile)
    t_out = therapy_agent(c_out, modality_scores=modality_scores, domain_profile=domain_profile)

    score_history.append(fused_score)
    modality_history.append(modality_scores.copy())
    m_out = monitoring_agent(score_history, modality_history=modality_history)

    return s_out, c_out, t_out, m_out


def _fuse_scores(modality_scores: dict) -> float:
    """Weighted fusion of available modality scores."""
    weights = {
        "face": 0.25,
        "behavior": 0.15,
        "questionnaire": 0.30,
        "eye_tracking": 0.15,
        "pose": 0.15,
    }
    total_w = 0.0
    weighted_sum = 0.0
    for mod, score in modality_scores.items():
        if score is not None and mod in weights:
            weighted_sum += weights[mod] * score
            total_w += weights[mod]
    if total_w == 0:
        return 0.5
    return float(np.clip(weighted_sum / total_w, 0, 1))


# ══════════════════════════════════════════════════════════════════════
#  IMAGE / VIDEO PROCESSING
# ══════════════════════════════════════════════════════════════════════

def _process_image(file_bytes: bytes) -> dict:
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_resized = pil_img.resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img_resized, dtype=np.float32), 0)

    face_score = 0.5
    if face_model is not None:
        pred = face_model.predict(img_array, verbose=0)[0][0]
        face_score = float(1.0 - pred)

    gradcam_b64 = None
    if face_model is not None:
        try:
            heatmap = make_gradcam_heatmap(img_array, face_model)
            overlay = overlay_gradcam(np.array(img_resized, dtype=np.uint8), heatmap, alpha=0.45)
            gradcam_b64 = numpy_to_base64_png(overlay)
        except Exception as e:
            print(f"Grad-CAM error: {e}")

    original_b64 = numpy_to_base64_png(np.array(img_resized, dtype=np.uint8))

    modality_scores = {"face": round(face_score, 4)}
    fused = _fuse_scores(modality_scores)
    s_out, c_out, t_out, m_out = _run_agent_pipeline(fused, modality_scores)

    return {
        "face_score": round(face_score, 4),
        "behavior_score": None,
        "fused_score": round(fused, 4),
        "modality_scores": modality_scores,
        "original_b64": original_b64,
        "gradcam_b64": gradcam_b64,
        "input_type": "image",
        "screening": s_out,
        "clinical": c_out,
        "therapy": t_out,
        "monitoring": m_out,
        "score_history": list(score_history),
    }


def _process_video(file_bytes: bytes, filename: str) -> dict:
    import cv2

    suffix = os.path.splitext(filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = 16
    frames = []
    if total > 0:
        indices = np.linspace(0, total - 1, max_frames, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, IMG_SIZE)
                frames.append(frame.astype(np.float32))
    cap.release()
    frames = np.array(frames) if frames else np.array([])

    face_score = 0.5
    if face_model is not None and len(frames) > 0:
        preds = face_model.predict(frames, verbose=0).flatten()
        face_score = float(1.0 - np.mean(preds))

    behavior_score = 0.5
    if behavior_model is not None and len(frames) > 0:
        feats = []
        for f in frames:
            inp = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(f, 0))
            feat = gap_layer(feature_extractor(inp, training=False))
            feats.append(feat.numpy().flatten())
        feat_seq = np.expand_dims(np.array(feats), 0)
        behavior_score = float(behavior_model.predict(feat_seq, verbose=0)[0][0])

    modality_scores = {
        "face": round(face_score, 4),
        "behavior": round(behavior_score, 4),
    }
    fused = _fuse_scores(modality_scores)

    gradcam_b64 = None
    original_b64 = None
    if len(frames) > 0:
        mid = frames[len(frames) // 2]
        original_b64 = numpy_to_base64_png(np.uint8(mid))
        if face_model is not None:
            try:
                heatmap = make_gradcam_heatmap(np.expand_dims(mid, 0), face_model)
                overlay = overlay_gradcam(np.uint8(mid), heatmap, alpha=0.45)
                gradcam_b64 = numpy_to_base64_png(overlay)
            except Exception as e:
                print(f"Grad-CAM error: {e}")

    os.unlink(tmp_path)

    s_out, c_out, t_out, m_out = _run_agent_pipeline(fused, modality_scores)

    return {
        "face_score": round(face_score, 4),
        "behavior_score": round(behavior_score, 4),
        "fused_score": round(fused, 4),
        "modality_scores": modality_scores,
        "original_b64": original_b64,
        "gradcam_b64": gradcam_b64,
        "input_type": "video",
        "screening": s_out,
        "clinical": c_out,
        "therapy": t_out,
        "monitoring": m_out,
        "score_history": list(score_history),
    }


# ══════════════════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════════

class QuestionnaireRequest(BaseModel):
    A1_Score: int = 0
    A2_Score: int = 0
    A3_Score: int = 0
    A4_Score: int = 0
    A5_Score: int = 0
    A6_Score: int = 0
    A7_Score: int = 0
    A8_Score: int = 0
    A9_Score: int = 0
    A10_Score: int = 0
    age: float = 5.0
    gender: int = 1       # 1=M, 0=F
    jundice: int = 0
    austim: int = 0


class FuseRequest(BaseModel):
    face: Optional[float] = None
    behavior: Optional[float] = None
    questionnaire: Optional[float] = None
    eye_tracking: Optional[float] = None
    pose: Optional[float] = None


# ══════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

@app.get("/api/status")
def status():
    return {
        "models": {
            "face_classifier": face_model is not None,
            "behavior_lstm": behavior_model is not None,
            "questionnaire_xgb": questionnaire_model is not None,
            "eye_tracking_xgb": eye_tracking_model is not None,
            "pose_skeleton_xgb": pose_model is not None,
            "cars_severity": cars_model is not None,
        },
        "feature_extractor": feature_extractor is not None,
        "sessions": len(score_history),
        "agents": {
            "screening": "v2 — Bayesian confidence + cross-modal attention",
            "clinical": "v2 — LLM-powered (GPT-4o-mini)" if is_llm_available()
                        else "v2 — enhanced template (set OPENAI_API_KEY for LLM)",
            "therapy": "v2 — RAG knowledge base" + (" + LLM personalization" if is_llm_available() else ""),
            "monitoring": "v2 — EWMA + linear forecast + CUSUM change-point",
        },
        "llm_available": is_llm_available(),
    }


@app.get("/api/model-info")
def model_info():
    info = {}
    if questionnaire_meta:
        info["questionnaire"] = {
            "accuracy": questionnaire_meta.get("accuracy"),
            "auc": questionnaire_meta.get("auc_roc"),
            "features": len(questionnaire_meta.get("features", [])),
            "domain_profile": questionnaire_meta.get("domain_profile"),
        }
    if eye_tracking_meta:
        info["eye_tracking"] = {
            "accuracy": eye_tracking_meta.get("test_accuracy"),
            "auc": eye_tracking_meta.get("test_auc"),
            "features": eye_tracking_meta.get("n_features"),
        }
    if pose_meta:
        info["pose_skeleton"] = {
            "accuracy": pose_meta.get("oof_accuracy"),
            "auc": pose_meta.get("oof_auc"),
            "features": pose_meta.get("n_features"),
            "participants": pose_meta.get("n_participants"),
        }
    if cars_meta:
        info["cars_severity"] = {
            "mae": cars_meta.get("loo_mae"),
            "r2": cars_meta.get("loo_r2"),
            "severity_accuracy": cars_meta.get("severity_accuracy"),
        }
    return info


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.filename is None:
        raise HTTPException(400, "No file provided")

    file_bytes = await file.read()
    filename = file.filename.lower()
    video_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    is_video = any(filename.endswith(ext) for ext in video_exts)

    result = _process_video(file_bytes, filename) if is_video else _process_image(file_bytes)

    save_result({
        "input": file.filename,
        "face_score": result["face_score"],
        "behavior_score": result.get("behavior_score"),
        "fused_score": result["fused_score"],
        "modality_scores": result["modality_scores"],
        "screening": result["screening"],
    })

    return result


@app.post("/api/questionnaire")
def questionnaire_analyze(req: QuestionnaireRequest):
    if questionnaire_model is None:
        raise HTTPException(503, "Questionnaire model not loaded")

    social_qs = [req.A1_Score, req.A2_Score, req.A4_Score, req.A6_Score, req.A10_Score]
    comm_qs = [req.A3_Score, req.A5_Score, req.A9_Score]
    behav_qs = [req.A7_Score, req.A8_Score]

    social_sum = sum(social_qs)
    communication_sum = sum(comm_qs)
    behavior_sum = sum(behav_qs)
    total_score = social_sum + communication_sum + behavior_sum

    features = np.array([[
        req.A1_Score, req.A2_Score, req.A3_Score, req.A4_Score, req.A5_Score,
        req.A6_Score, req.A7_Score, req.A8_Score, req.A9_Score, req.A10_Score,
        req.age, req.gender, req.jundice, req.austim,
        social_sum, communication_sum, behavior_sum, total_score,
    ]], dtype=np.float32)

    if questionnaire_scaler is not None:
        features = questionnaire_scaler.transform(features)

    prob = float(questionnaire_model.predict_proba(features)[0][1])

    domain_total = max(social_sum + communication_sum + behavior_sum, 1)
    domain_profile = {
        "Social": round(social_sum / domain_total, 3),
        "Communication": round(communication_sum / domain_total, 3),
        "Behavior": round(behavior_sum / domain_total, 3),
    }

    modality_scores = {"questionnaire": round(prob, 4)}
    fused = _fuse_scores(modality_scores)
    s_out, c_out, t_out, m_out = _run_agent_pipeline(fused, modality_scores, domain_profile)

    return {
        "questionnaire_score": round(prob, 4),
        "fused_score": round(fused, 4),
        "domain_profile": domain_profile,
        "domain_scores": {
            "social_sum": social_sum,
            "communication_sum": communication_sum,
            "behavior_sum": behavior_sum,
            "total_score": total_score,
        },
        "modality_scores": modality_scores,
        "screening": s_out,
        "clinical": c_out,
        "therapy": t_out,
        "monitoring": m_out,
        "score_history": list(score_history),
    }


@app.post("/api/fuse")
def fuse_modalities(req: FuseRequest):
    modality_scores = {}
    if req.face is not None:
        modality_scores["face"] = req.face
    if req.behavior is not None:
        modality_scores["behavior"] = req.behavior
    if req.questionnaire is not None:
        modality_scores["questionnaire"] = req.questionnaire
    if req.eye_tracking is not None:
        modality_scores["eye_tracking"] = req.eye_tracking
    if req.pose is not None:
        modality_scores["pose"] = req.pose

    if not modality_scores:
        raise HTTPException(400, "At least one modality score required")

    fused = _fuse_scores(modality_scores)
    s_out, c_out, t_out, m_out = _run_agent_pipeline(fused, modality_scores)

    return {
        "fused_score": round(fused, 4),
        "modality_scores": modality_scores,
        "screening": s_out,
        "clinical": c_out,
        "therapy": t_out,
        "monitoring": m_out,
        "score_history": list(score_history),
    }


@app.get("/api/history")
def get_history():
    return {
        "score_history": list(score_history),
        "modality_history": list(modality_history),
        "sessions": len(score_history),
    }


@app.post("/api/clear")
def clear_history():
    score_history.clear()
    modality_history.clear()
    return {"status": "cleared"}


# ── Run directly ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n  → API docs at http://localhost:8000/docs\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
