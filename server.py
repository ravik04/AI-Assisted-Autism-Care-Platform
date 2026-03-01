"""
Autism AI Screening — Flask Backend API
Serves the HTML UI and provides /api/analyze endpoint.
"""

import os
import sys
import io
import base64
import uuid
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Project imports ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from agents.screening_agent import screening_agent
from agents.clinical_agent import clinical_agent
from agents.therapy_agent import therapy_agent
from agents.monitoring_agent import monitoring_agent
from utils.gradcam import make_gradcam_heatmap, overlay_gradcam
from utils.logger import save_result

# ── Paths ──────────────────────────────────────────────────────────────
FACE_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "face_classifier.keras")
BEH_MODEL_PATH  = os.path.join(BASE_DIR, "saved_models", "behavior_lstm.keras")
STATIC_DIR      = os.path.join(BASE_DIR, "static")
IMG_SIZE = (224, 224)

# ── Flask app ──────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)

# ── Load models once at startup ───────────────────────────────────────
print("Loading models...")
face_model = None
behavior_model = None
feature_extractor = None
gap_layer = None

if os.path.exists(FACE_MODEL_PATH):
    face_model = tf.keras.models.load_model(FACE_MODEL_PATH)
    print("  ✓ Face classifier loaded")
else:
    print("  ⚠ Face classifier not found")

if os.path.exists(BEH_MODEL_PATH):
    behavior_model = tf.keras.models.load_model(BEH_MODEL_PATH)
    print("  ✓ Behavior LSTM loaded")
else:
    print("  ⚠ Behavior LSTM not found")

# MobileNetV2 feature extractor for behavior model
feature_extractor = tf.keras.applications.MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
gap_layer = tf.keras.layers.GlobalAveragePooling2D()
print("  ✓ MobileNetV2 feature extractor loaded")

# In-memory score history (per-session via simple list)
score_history = []


# ── Helper functions ───────────────────────────────────────────────────
def img_to_base64(img_array):
    """Convert numpy RGB array to base64 PNG string."""
    img = Image.fromarray(np.uint8(img_array))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def process_image(file_bytes):
    """Process an uploaded image file → scores + gradcam."""
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_resized = pil_img.resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img_resized, dtype=np.float32), 0)

    # Face score
    if face_model is not None:
        pred = face_model.predict(img_array, verbose=0)[0][0]
        face_score = float(1.0 - pred)
    else:
        face_score = 0.5

    # Grad-CAM
    gradcam_b64 = None
    if face_model is not None:
        try:
            heatmap = make_gradcam_heatmap(img_array, face_model)
            original_rgb = np.array(img_resized, dtype=np.uint8)
            overlay = overlay_gradcam(original_rgb, heatmap, alpha=0.45)
            gradcam_b64 = img_to_base64(overlay)
        except Exception as e:
            print(f"Grad-CAM error: {e}")

    # Original image as base64
    original_b64 = img_to_base64(np.array(img_resized, dtype=np.uint8))

    return {
        "face_score": face_score,
        "behavior_score": None,
        "fused_score": face_score,
        "original_b64": original_b64,
        "gradcam_b64": gradcam_b64,
        "input_type": "image",
    }


def process_video(file_bytes, filename):
    """Process an uploaded video file → scores + gradcam on middle frame."""
    import cv2
    import tempfile

    suffix = os.path.splitext(filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # Extract frames
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

    # Face score from frames
    if face_model is not None and len(frames) > 0:
        preds = face_model.predict(frames, verbose=0).flatten()
        face_score = float(1.0 - np.mean(preds))
    else:
        face_score = 0.5

    # Behavior score
    if behavior_model is not None and len(frames) > 0:
        feats = []
        for f in frames:
            inp = tf.keras.applications.mobilenet_v2.preprocess_input(
                np.expand_dims(f, 0)
            )
            feat = gap_layer(feature_extractor(inp, training=False))
            feats.append(feat.numpy().flatten())
        feat_seq = np.expand_dims(np.array(feats), 0)
        behavior_score = float(behavior_model.predict(feat_seq, verbose=0)[0][0])
    else:
        behavior_score = 0.5

    fused = float(np.clip(0.6 * face_score + 0.4 * behavior_score, 0, 1))

    # Grad-CAM on middle frame
    gradcam_b64 = None
    original_b64 = None
    if len(frames) > 0:
        mid = frames[len(frames) // 2]
        mid_arr = np.expand_dims(mid, 0)
        original_b64 = img_to_base64(np.uint8(mid))
        if face_model is not None:
            try:
                heatmap = make_gradcam_heatmap(mid_arr, face_model)
                overlay = overlay_gradcam(np.uint8(mid), heatmap, alpha=0.45)
                gradcam_b64 = img_to_base64(overlay)
            except Exception as e:
                print(f"Grad-CAM error: {e}")

    os.unlink(tmp_path)

    return {
        "face_score": face_score,
        "behavior_score": behavior_score,
        "fused_score": fused,
        "original_b64": original_b64,
        "gradcam_b64": gradcam_b64,
        "input_type": "video",
    }


# ── Routes ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename.lower()
    file_bytes = file.read()

    video_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    is_video = any(filename.endswith(ext) for ext in video_exts)

    if is_video:
        result = process_video(file_bytes, filename)
    else:
        result = process_image(file_bytes)

    # Multi-agent chain
    final_score = result["fused_score"]
    score_history.append(final_score)

    s_out = screening_agent(final_score)
    c_out = clinical_agent(s_out)
    t_out = therapy_agent(c_out)
    m_out = monitoring_agent(score_history)

    result.update({
        "screening": s_out,
        "clinical": c_out,
        "therapy": t_out,
        "monitoring": m_out,
        "score_history": score_history.copy(),
    })

    # Save to disk
    save_result({
        "input": file.filename,
        "face_score": result["face_score"],
        "behavior_score": result["behavior_score"],
        "fused_score": result["fused_score"],
        "screening": s_out,
        "clinical": c_out,
        "therapy": t_out,
        "monitoring": m_out,
    })

    return jsonify(result)


@app.route("/api/history", methods=["GET"])
def get_history():
    return jsonify({"score_history": score_history})


@app.route("/api/clear", methods=["POST"])
def clear_history():
    score_history.clear()
    return jsonify({"status": "cleared"})


# ── Run ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n  → Open http://localhost:5000 in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
