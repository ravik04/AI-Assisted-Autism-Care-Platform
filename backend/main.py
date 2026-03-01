"""
Autism AI Screening — FastAPI Backend (v5 Full-Stack Multi-Modal)
================================================================
REST API serving 8 trained models + 4 cooperative agents + consent + RLHF.

Models:
  1. Face Classifier      (MobileNetV2, .keras)
  2. Behavior LSTM        (MobileNetV2+LSTM, .keras)
  3. Questionnaire XGB    (XGBoost, .pkl)
  4. Eye-Tracking XGB     (XGBoost, .pkl)
  5. Pose/Skeleton XGB    (XGBoost, .pkl)
  6. CARS Severity Ridge  (Ridge regression, .pkl) — enrichment
  7. Audio/Speech CNN+GRU (.keras) — prosody & vocalization
  8. EEG Neural 1D-CNN    (.keras) — brainwave patterns
  9. Attention Fusion     (.keras) — learned multi-modal fusion

Endpoints:
  POST /api/analyze           — image/video → face + behavior scores + agents
  POST /api/questionnaire     — screening questionnaire → risk score + agents
  POST /api/analyze-audio     — audio file → speech/prosody analysis
  POST /api/analyze-eeg       — EEG CSV file → neural signal analysis
  POST /api/fuse              — combine all modality scores → full pipeline
  GET  /api/history           — score history
  POST /api/clear             — reset history
  GET  /api/status            — model status
  GET  /api/model-info        — metadata for all models
  POST /api/consent           — grant consent
  GET  /api/consent/{child_id} — get consent status
  DELETE /api/consent/{child_id} — revoke consent
  POST /api/feedback          — submit RLHF feedback
  GET  /api/feedback/summary  — aggregate feedback stats
  POST /api/children          — create child profile
  GET  /api/children          — list all children
  GET  /api/children/{id}     — get child profile
  GET  /api/explain           — get screening explanation
"""

import os, sys, io, json, base64, tempfile, hashlib, secrets
from datetime import datetime, timedelta, timezone
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, Dict, List
import jwt as pyjwt

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
from utils.consent import grant_consent, revoke_consent, get_consent_summary, check_consent, CONSENT_CATEGORIES
from utils.feedback import submit_feedback, get_feedback_summary, get_reward_signal
from utils.storage import create_child, get_child, list_children, update_child, save_session, get_sessions, get_longitudinal_data
from utils.explainability import explain_screening_result, compute_feature_importance
from utils.audio_features import extract_audio_features, N_FEATURES as AUDIO_N_FEATURES
from utils.eeg_features import extract_eeg_features, N_FEATURES as EEG_N_FEATURES
from models.attention_fusion import (
    prepare_fusion_input, fallback_fusion, MODALITY_NAMES,
    ModalityEmbedding, MaskAndMultiply, CrossModalAttention, WeightedFusionAggregation
)

# ── Paths ──────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(PROJECT_DIR, "saved_models")
IMG_SIZE = (224, 224)

# ── FastAPI App ────────────────────────────────────────────────────────
app = FastAPI(
    title="Autism AI Multi-Modal Screening API",
    description="8-model, 4-agent developmental screening pipeline with consent & RLHF",
    version="5.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════
#  JWT AUTHENTICATION
# ══════════════════════════════════════════════════════════════════════
JWT_SECRET = os.environ.get("JWT_SECRET", "autism-care-ai-secret-key-change-in-prod")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 72

# Simple JSON-based user store (for prototype)
USERS_FILE = os.path.join(PROJECT_DIR, "data", "users.json")

def _load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}

def _save_users(users):
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def _create_token(user_id: str, name: str, email: str, role: str) -> str:
    payload = {
        "user_id": user_id,
        "name": name,
        "email": email,
        "role": role,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return pyjwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Optional JWT auth — returns user dict or None."""
    if not credentials:
        return None
    try:
        payload = pyjwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


class AuthLoginRequest(BaseModel):
    email: str
    password: str

class AuthRegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    role: str = "parent"


@app.post("/api/auth/register")
def auth_register(req: AuthRegisterRequest):
    """Register a new user."""
    users = _load_users()
    if req.email in users:
        raise HTTPException(status_code=400, detail="Email already registered")
    if req.role not in ("parent", "clinician", "therapist"):
        raise HTTPException(status_code=400, detail="Invalid role")
    user_id = secrets.token_hex(4)
    users[req.email] = {
        "id": user_id,
        "name": req.name,
        "email": req.email,
        "role": req.role,
        "password_hash": _hash_password(req.password),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_users(users)
    token = _create_token(user_id, req.name, req.email, req.role)
    return {"token": token, "user": {"id": user_id, "name": req.name, "email": req.email, "role": req.role}}


@app.post("/api/auth/login")
def auth_login(req: AuthLoginRequest):
    """Authenticate and return JWT."""
    users = _load_users()
    user = users.get(req.email)
    if not user or user["password_hash"] != _hash_password(req.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = _create_token(user["id"], user["name"], user["email"], user["role"])
    return {"token": token, "user": {"id": user["id"], "name": user["name"], "email": user["email"], "role": user["role"]}}


@app.get("/api/auth/me")
def auth_me(current_user=Depends(get_current_user)):
    """Get current user from JWT."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"user_id": current_user["user_id"], "name": current_user["name"], "email": current_user["email"], "role": current_user["role"]}


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
audio_model = None
audio_scaler = None
audio_meta = None
eeg_model = None
eeg_scaler = None
eeg_meta = None
fusion_model = None
fusion_meta = None


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
    global audio_model, audio_scaler, audio_meta
    global eeg_model, eeg_scaler, eeg_meta
    global fusion_model, fusion_meta

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

    # 7) Audio/Speech CNN+GRU
    ap = os.path.join(MODELS_DIR, "audio_speech_model.keras")
    if os.path.exists(ap):
        audio_model = tf.keras.models.load_model(ap)
        print("  ✓ Audio/Speech CNN+GRU loaded")
    else:
        print("  ✗ Audio/Speech model not found")
    audio_scaler = _load_pkl("audio_speech_scaler.pkl")
    audio_meta = _load_json("audio_speech_metadata.json")

    # 8) EEG Neural 1D-CNN
    ep = os.path.join(MODELS_DIR, "eeg_neural_model.keras")
    if os.path.exists(ep):
        eeg_model = tf.keras.models.load_model(ep)
        print("  ✓ EEG Neural 1D-CNN loaded")
    else:
        print("  ✗ EEG Neural model not found")
    eeg_scaler = _load_pkl("eeg_neural_scaler.pkl")
    eeg_meta = _load_json("eeg_neural_metadata.json")

    # 9) Attention Fusion
    afp = os.path.join(MODELS_DIR, "attention_fusion.keras")
    if os.path.exists(afp):
        fusion_model = tf.keras.models.load_model(afp)
        print("  ✓ Attention Fusion loaded")
    else:
        print("  ✗ Attention Fusion model not found (using fallback weights)")
    fusion_meta = _load_json("attention_fusion_metadata.json")

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
    """Learned attention fusion with fallback to weighted averaging."""
    if fusion_model is not None:
        try:
            x = prepare_fusion_input(modality_scores)
            pred = fusion_model.predict(x, verbose=0)[0][0]
            return float(np.clip(pred, 0, 1))
        except Exception as e:
            print(f"Fusion model error, using fallback: {e}")
    return fallback_fusion(modality_scores)


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
    audio: Optional[float] = None
    eeg: Optional[float] = None


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
            "audio_speech": audio_model is not None,
            "eeg_neural": eeg_model is not None,
            "attention_fusion": fusion_model is not None,
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
        "features": {
            "consent_management": True,
            "rlhf_feedback": True,
            "child_profiles": True,
            "explainability": True,
            "attention_fusion": fusion_model is not None,
            "audio_analysis": audio_model is not None,
            "eeg_analysis": eeg_model is not None,
            "neurodiversity_affirming": True,
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
    if audio_meta:
        info["audio_speech"] = {
            "architecture": audio_meta.get("architecture"),
            "accuracy": audio_meta.get("accuracy"),
            "n_features": audio_meta.get("n_features"),
        }
    if eeg_meta:
        info["eeg_neural"] = {
            "architecture": eeg_meta.get("architecture"),
            "accuracy": eeg_meta.get("accuracy"),
            "auc": eeg_meta.get("auc"),
            "n_features": eeg_meta.get("n_features"),
        }
    if fusion_meta:
        info["attention_fusion"] = {
            "architecture": fusion_meta.get("architecture"),
            "accuracy": fusion_meta.get("accuracy"),
            "auc": fusion_meta.get("auc"),
            "modalities": fusion_meta.get("modalities"),
            "handles_missing": fusion_meta.get("handles_missing_modalities"),
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
    if req.audio is not None:
        modality_scores["audio"] = req.audio
    if req.eeg is not None:
        modality_scores["eeg"] = req.eeg

    if not modality_scores:
        raise HTTPException(400, "At least one modality score required")

    fused = _fuse_scores(modality_scores)
    s_out, c_out, t_out, m_out = _run_agent_pipeline(fused, modality_scores)

    # Explainability
    explanation = explain_screening_result(
        modality_scores, fused,
        s_out.get("risk_level", "moderate") if isinstance(s_out, dict) else "moderate",
    )
    feature_importance = compute_feature_importance(modality_scores)

    return {
        "fused_score": round(fused, 4),
        "modality_scores": modality_scores,
        "screening": s_out,
        "clinical": c_out,
        "therapy": t_out,
        "monitoring": m_out,
        "explanation": explanation,
        "feature_importance": feature_importance,
        "score_history": list(score_history),
    }


# ══════════════════════════════════════════════════════════════════════
#  MULTI-FILE UPLOAD (for Parent multi-image/video)
# ══════════════════════════════════════════════════════════════════════

@app.post("/api/analyze-multi")
async def analyze_multiple(files: List[UploadFile] = File(...)):
    """Analyze multiple images/videos and return aggregated results."""
    if not files:
        raise HTTPException(400, "No files provided")
    if len(files) > 10:
        raise HTTPException(400, "Maximum 10 files allowed")

    all_results = []
    face_scores = []
    behavior_scores = []

    for f in files:
        file_bytes = await f.read()
        try:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize(IMG_SIZE)
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)
            is_video = False
        except Exception:
            is_video = True
            img_array = None

        file_result = {"filename": f.filename, "type": "video" if is_video else "image"}

        if not is_video and face_model is not None:
            face_pred = float(face_model.predict(img_array, verbose=0)[0][0])
            file_result["face_score"] = face_pred
            face_scores.append(face_pred)

        if not is_video and behavior_model is not None:
            try:
                frames = np.stack([img_array[0]] * 10)
                if feature_extractor and gap_layer:
                    feats = gap_layer.predict(feature_extractor.predict(frames, verbose=0), verbose=0)
                    seq = np.expand_dims(feats, axis=0)
                    beh_pred = float(behavior_model.predict(seq, verbose=0)[0][0])
                    file_result["behavior_score"] = beh_pred
                    behavior_scores.append(beh_pred)
            except Exception:
                pass

        all_results.append(file_result)

    # Aggregate
    avg_face = float(np.mean(face_scores)) if face_scores else None
    avg_behavior = float(np.mean(behavior_scores)) if behavior_scores else None

    modalities = {}
    if avg_face is not None:
        modalities["face"] = avg_face
    if avg_behavior is not None:
        modalities["behavior"] = avg_behavior

    # Run agents on aggregated scores
    fused = np.mean([v for v in modalities.values() if v is not None]) if modalities else 0.5
    score_history.append(fused)
    modality_history.append(modalities)

    screening = screening_agent(fused, modalities)
    clinical = clinical_agent(screening, modalities)
    therapy = therapy_agent(clinical, modalities)
    monitoring = monitoring_agent(list(score_history), list(modality_history))

    return {
        "per_file_results": all_results,
        "aggregated_face_score": avg_face,
        "aggregated_behavior_score": avg_behavior,
        "fused_score": fused,
        "modality_scores": modalities,
        "screening": screening,
        "clinical": clinical,
        "therapy": therapy,
        "monitoring": monitoring,
        "files_processed": len(all_results),
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


# ══════════════════════════════════════════════════════════════════════
#  AUDIO / SPEECH ANALYSIS
# ══════════════════════════════════════════════════════════════════════

@app.post("/api/analyze-audio")
async def analyze_audio(file: UploadFile = File(None), use_synthetic: bool = True):
    """Analyze audio/speech for autism-related prosody markers."""
    result = None

    if file is not None:
        file_bytes = await file.read()
        try:
            result = extract_audio_features(file_bytes)
        except Exception as e:
            print(f"Audio feature extraction error: {e}")

    if result is None and use_synthetic:
        from utils.audio_features import _synthetic_features
        result = _synthetic_features()

    if result is None:
        raise HTTPException(400, "Could not extract audio features. Upload a WAV file or set use_synthetic=true.")

    features = np.asarray(result["features"], dtype=np.float32)
    prosody = result.get("prosody_summary", {})

    audio_score = 0.5
    if audio_model is not None and audio_scaler is not None:
        scaled = audio_scaler.transform(features.reshape(1, -1))
        # Reshape for CNN+GRU: (1, timesteps, features) — repeat to create sequence
        n_timesteps = 20
        seq = np.tile(scaled, (1, n_timesteps, 1)).reshape(1, n_timesteps, -1)
        pred = audio_model.predict(seq, verbose=0)[0][0]
        audio_score = float(np.clip(pred, 0, 1))
    elif audio_model is not None:
        seq = np.tile(features.reshape(1, -1), (1, 20, 1)).reshape(1, 20, -1).astype(np.float32)
        pred = audio_model.predict(seq, verbose=0)[0][0]
        audio_score = float(np.clip(pred, 0, 1))

    modality_scores = {"audio": round(audio_score, 4)}
    fused = _fuse_scores(modality_scores)
    s_out, c_out, t_out, m_out = _run_agent_pipeline(fused, modality_scores)

    return {
        "audio_score": round(audio_score, 4),
        "fused_score": round(fused, 4),
        "modality_scores": modality_scores,
        "feature_summary": {
            "n_features": len(features),
            "pitch_variability": prosody.get("pitch_variability", 0),
            "pause_ratio": prosody.get("pause_ratio_pct", 0),
            "speech_rate": prosody.get("speech_rate_syl_per_sec", 0),
        },
        "screening": s_out,
        "clinical": c_out,
        "therapy": t_out,
        "monitoring": m_out,
        "score_history": list(score_history),
    }


# ══════════════════════════════════════════════════════════════════════
#  EEG / NEURAL SIGNAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════

@app.post("/api/analyze-eeg")
async def analyze_eeg(file: UploadFile = File(None), use_synthetic: bool = True):
    """Analyze EEG data for autism-related neural signatures."""
    result = None

    if file is not None:
        file_bytes = await file.read()
        try:
            # Save temp CSV and extract
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            result = extract_eeg_features(tmp_path)
            os.unlink(tmp_path)
        except Exception as e:
            print(f"EEG feature extraction error: {e}")

    if result is None and use_synthetic:
        from utils.eeg_features import _synthetic_eeg_features
        result = _synthetic_eeg_features()

    if result is None:
        raise HTTPException(400, "Could not extract EEG features. Upload a CSV file or set use_synthetic=true.")

    features = np.asarray(result["features"], dtype=np.float32)
    band_powers = result.get("band_powers", {})

    eeg_score = 0.5
    if eeg_model is not None and eeg_scaler is not None:
        scaled = eeg_scaler.transform(features.reshape(1, -1))
        inp = scaled.reshape(1, -1, 1).astype(np.float32)
        pred = eeg_model.predict(inp, verbose=0)[0][0]
        eeg_score = float(np.clip(pred, 0, 1))
    elif eeg_model is not None:
        inp = features.reshape(1, -1, 1).astype(np.float32)
        pred = eeg_model.predict(inp, verbose=0)[0][0]
        eeg_score = float(np.clip(pred, 0, 1))

    modality_scores = {"eeg": round(eeg_score, 4)}
    fused = _fuse_scores(modality_scores)
    s_out, c_out, t_out, m_out = _run_agent_pipeline(fused, modality_scores)

    return {
        "eeg_score": round(eeg_score, 4),
        "fused_score": round(fused, 4),
        "modality_scores": modality_scores,
        "feature_summary": {
            "n_features": len(features),
            "theta_power": band_powers.get("theta", 0),
            "alpha_power": band_powers.get("alpha", 0),
            "theta_beta_ratio": round(float(features[5]) if len(features) > 5 else 0, 4),
        },
        "screening": s_out,
        "clinical": c_out,
        "therapy": t_out,
        "monitoring": m_out,
        "score_history": list(score_history),
    }


# ══════════════════════════════════════════════════════════════════════
#  CONSENT MANAGEMENT
# ══════════════════════════════════════════════════════════════════════

class ConsentRequest(BaseModel):
    child_id: str
    guardian_name: str
    guardian_email: Optional[str] = None
    categories: List[str] = []

@app.post("/api/consent")
def create_consent(req: ConsentRequest):
    """Grant consent for data processing categories."""
    record = grant_consent(
        child_id=req.child_id,
        guardian_name=req.guardian_name,
        categories=req.categories if req.categories else CONSENT_CATEGORIES,
        guardian_email=req.guardian_email,
    )
    return record

@app.get("/api/consent/{child_id}")
def get_consent_status(child_id: str):
    """Get current consent status for a child."""
    return get_consent_summary(child_id)

@app.delete("/api/consent/{child_id}")
def delete_consent(child_id: str):
    """Revoke all consent for a child."""
    return revoke_consent(child_id)


# ══════════════════════════════════════════════════════════════════════
#  RLHF FEEDBACK
# ══════════════════════════════════════════════════════════════════════

class FeedbackRequest(BaseModel):
    session_id: str
    feedback_type: str = "screening"
    rating: str = "neutral"
    comment: Optional[str] = None
    child_id: Optional[str] = None
    user_role: str = "clinician"
    corrections: Optional[Dict] = None

@app.post("/api/feedback")
def post_feedback(req: FeedbackRequest):
    """Submit feedback on AI recommendations for RLHF."""
    record = submit_feedback(
        session_id=req.session_id,
        feedback_type=req.feedback_type,
        rating=req.rating,
        comment=req.comment,
        child_id=req.child_id,
        user_role=req.user_role,
        corrections=req.corrections,
    )
    return record

@app.get("/api/feedback/summary")
def feedback_summary():
    """Get aggregated feedback statistics."""
    return get_feedback_summary()


# ══════════════════════════════════════════════════════════════════════
#  CHILD PROFILES
# ══════════════════════════════════════════════════════════════════════

class ChildRequest(BaseModel):
    name: str
    age_months: int
    guardian_name: str
    notes: Optional[str] = None

@app.post("/api/children")
def create_child_profile(req: ChildRequest):
    """Create a new child profile."""
    return create_child(
        name=req.name,
        age_months=req.age_months,
        guardian_name=req.guardian_name,
        notes=req.notes,
    )

@app.get("/api/children")
def list_all_children():
    """List all child profiles."""
    return list_children()

@app.get("/api/children/{child_id}")
def get_child_profile(child_id: str):
    """Get a child profile by ID."""
    child = get_child(child_id)
    if child is None:
        raise HTTPException(404, "Child not found")
    sessions = get_sessions(child_id)
    longitudinal = get_longitudinal_data(child_id)
    return {**child, "sessions": sessions, "longitudinal": longitudinal}


# ══════════════════════════════════════════════════════════════════════
#  EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════

class ExplainRequest(BaseModel):
    modality_scores: Dict[str, float]
    fused_score: float
    risk_level: str = "moderate"
    child_age_months: Optional[int] = None

@app.post("/api/explain")
def explain_result(req: ExplainRequest):
    """Get detailed explainability report for a screening result."""
    explanation = explain_screening_result(
        modality_scores=req.modality_scores,
        fused_score=req.fused_score,
        risk_level=req.risk_level,
        child_age_months=req.child_age_months,
    )
    importance = compute_feature_importance(req.modality_scores)
    return {"explanation": explanation, "feature_importance": importance}


# ── Serve frontend (React SPA build, fallback to old vanilla) ──────────
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

REACT_DIR = os.path.join(PROJECT_DIR, "frontend-react", "dist")
LEGACY_DIR = os.path.join(PROJECT_DIR, "frontend")
FRONTEND_DIR = REACT_DIR if os.path.isdir(REACT_DIR) else LEGACY_DIR

if os.path.isdir(FRONTEND_DIR):
    # Serve static assets (js, css, images, etc.)
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets") if os.path.isdir(os.path.join(FRONTEND_DIR, "assets")) else None

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve React SPA – return index.html for all non-API routes."""
        file_path = os.path.join(FRONTEND_DIR, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ── Run directly ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"\n  → API docs at http://localhost:{port}/docs\n")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
