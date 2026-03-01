"""
Autism AI Screening — Streamlit Web UI
Upload an image or video → get multi-agent screening results + Grad-CAM.
"""

import os
import sys
import tempfile
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# ── Make project imports work ──────────────────────────────────────────
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
BEH_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "behavior_lstm.keras")
IMG_SIZE = (224, 224)

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Autism AI Screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached model loading ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading face classifier...")
def load_face_model():
    if os.path.exists(FACE_MODEL_PATH):
        return tf.keras.models.load_model(FACE_MODEL_PATH)
    return None


@st.cache_resource(show_spinner="Loading behavior LSTM...")
def load_behavior_model():
    if os.path.exists(BEH_MODEL_PATH):
        return tf.keras.models.load_model(BEH_MODEL_PATH)
    return None


@st.cache_resource(show_spinner="Loading MobileNetV2 feature extractor...")
def load_feature_extractor():
    base = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    gap = tf.keras.layers.GlobalAveragePooling2D()
    return base, gap


# ── Scoring helpers ────────────────────────────────────────────────────
def score_face_image(img_array, model):
    """Return P(autistic) for a (1,224,224,3) array."""
    pred = model.predict(img_array, verbose=0)[0][0]
    return float(1.0 - pred)


def extract_video_frames(video_path, max_frames=16):
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return np.array([])
    indices = np.linspace(0, total - 1, max_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, IMG_SIZE)
            frames.append(frame.astype(np.float32))
    cap.release()
    return np.array(frames)


def score_behavior_video(video_path, beh_model):
    frames = extract_video_frames(video_path, max_frames=16)
    if len(frames) == 0:
        return 0.5
    extractor, gap = load_feature_extractor()
    feats = []
    for f in frames:
        inp = tf.keras.applications.mobilenet_v2.preprocess_input(
            np.expand_dims(f, 0)
        )
        feat = gap(extractor(inp, training=False))
        feats.append(feat.numpy().flatten())
    feat_seq = np.expand_dims(np.array(feats), 0)
    return float(beh_model.predict(feat_seq, verbose=0)[0][0])


def fuse_scores(face_score, behavior_score):
    return float(np.clip(0.6 * face_score + 0.4 * behavior_score, 0, 1))


# ── State ──────────────────────────────────────────────────────────────
if "score_history" not in st.session_state:
    st.session_state.score_history = []

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/brain--v2.png",
        width=64,
    )
    st.title("Autism AI Screening")
    st.caption("V3 — Multi-Agent Pipeline")
    st.markdown("---")

    st.subheader("Upload Input")
    input_type = st.radio("Input type", ["Image", "Video"], horizontal=True)

    if input_type == "Image":
        uploaded = st.file_uploader(
            "Upload a facial image", type=["jpg", "jpeg", "png", "bmp", "webp"]
        )
    else:
        uploaded = st.file_uploader(
            "Upload a video clip", type=["mp4", "avi", "mov", "mkv", "webm"]
        )

    st.markdown("---")
    st.subheader("Model Status")
    face_model = load_face_model()
    beh_model = load_behavior_model()
    st.write("Face Classifier:", "✅ Loaded" if face_model else "❌ Missing")
    st.write("Behavior LSTM:", "✅ Loaded" if beh_model else "❌ Missing")

    st.markdown("---")
    st.subheader("Architecture")
    st.markdown(
        """
        ```
        MobileNetV2 (face)
            ↓
        Screening Agent
            ↓
        Clinical Agent
            ↓
        Therapy Agent
            ↓
        Monitoring Agent
        ```
        """
    )

    if st.button("Clear Score History"):
        st.session_state.score_history = []
        st.rerun()


# ── Risk colour helpers ────────────────────────────────────────────────
RISK_COLORS = {
    "LOW_RISK": "#2ecc71",
    "MONITOR": "#f39c12",
    "CLINICAL_REVIEW": "#e74c3c",
}

RISK_LABELS = {
    "LOW_RISK": "Low Risk",
    "MONITOR": "Monitor",
    "CLINICAL_REVIEW": "Clinical Review",
}


def risk_badge(state):
    color = RISK_COLORS.get(state, "#888")
    label = RISK_LABELS.get(state, state)
    return f'<span style="background:{color};color:white;padding:4px 16px;border-radius:12px;font-weight:bold;font-size:1.1em;">{label}</span>'


# ── Main area ──────────────────────────────────────────────────────────
st.title("🧠 Autism AI Screening Dashboard")

if uploaded is None:
    st.info(
        "👈 Upload an image or video from the sidebar to begin screening."
    )
    # Show score history if any
    if st.session_state.score_history:
        st.subheader("📈 Score History")
        st.line_chart(st.session_state.score_history)
    st.stop()

# ── Process upload ─────────────────────────────────────────────────────
with st.spinner("Analyzing..."):
    if input_type == "Image":
        # Load & display
        pil_img = Image.open(uploaded).convert("RGB")
        img_resized = pil_img.resize(IMG_SIZE)
        img_array = np.expand_dims(np.array(img_resized, dtype=np.float32), 0)

        # Face score
        if face_model:
            f_score = score_face_image(img_array, face_model)
        else:
            f_score = 0.5
        b_score = None
        final_score = f_score

    else:  # Video
        # Save to temp file for OpenCV
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        frames = extract_video_frames(tmp_path)
        if face_model and len(frames) > 0:
            preds = face_model.predict(frames, verbose=0).flatten()
            f_score = float(1.0 - np.mean(preds))
        else:
            f_score = 0.5

        if beh_model:
            b_score = score_behavior_video(tmp_path, beh_model)
        else:
            b_score = 0.5

        final_score = fuse_scores(f_score, b_score)

        # Use middle frame for display
        if len(frames) > 0:
            pil_img = Image.fromarray(np.uint8(frames[len(frames) // 2]))
            img_resized = pil_img
            img_array = np.expand_dims(np.array(img_resized, dtype=np.float32), 0)
        else:
            pil_img = None
            img_array = None

        os.unlink(tmp_path)

# Run multi-agent chain
st.session_state.score_history.append(final_score)
s_out = screening_agent(final_score)
c_out = clinical_agent(s_out)
t_out = therapy_agent(c_out)
m_out = monitoring_agent(st.session_state.score_history)

state = s_out["state"]

# ── Layout ─────────────────────────────────────────────────────────────
col_img, col_cam = st.columns(2)

with col_img:
    st.subheader("Input")
    if pil_img:
        st.image(pil_img, caption=uploaded.name, use_container_width=True)

with col_cam:
    st.subheader("Grad-CAM Heatmap")
    if face_model and img_array is not None:
        try:
            heatmap = make_gradcam_heatmap(img_array, face_model)
            original_rgb = np.array(img_resized, dtype=np.uint8)
            overlay = overlay_gradcam(original_rgb, heatmap, alpha=0.45)
            st.image(overlay, caption="Model attention regions", use_container_width=True)
        except Exception as e:
            st.warning(f"Grad-CAM failed: {e}")
    else:
        st.info("Grad-CAM requires the face classifier model.")

# ── Scores ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"### Risk Assessment &nbsp; {risk_badge(state)}", unsafe_allow_html=True)

score_cols = st.columns(4 if b_score is not None else 3)

with score_cols[0]:
    st.metric("Face Score", f"{f_score:.3f}")
if b_score is not None:
    with score_cols[1]:
        st.metric("Behavior Score", f"{b_score:.3f}")
    with score_cols[2]:
        st.metric("Fused Score", f"{final_score:.3f}")
    with score_cols[3]:
        st.metric("Trend", f"{m_out['trend']:.3f}")
else:
    with score_cols[1]:
        st.metric("Final Score", f"{final_score:.3f}")
    with score_cols[2]:
        st.metric("Trend", f"{m_out['trend']:.3f}")

# ── Agent Outputs ──────────────────────────────────────────────────────
st.markdown("---")
agent_tabs = st.tabs(
    ["🔍 Screening", "🏥 Clinical", "💊 Therapy", "📊 Monitoring"]
)

with agent_tabs[0]:
    st.json(s_out)

with agent_tabs[1]:
    st.markdown(f"**Clinical Note:** {c_out['clinical_note']}")
    st.json(c_out)

with agent_tabs[2]:
    st.markdown("**Recommended Therapy Plan:**")
    for i, plan in enumerate(t_out["therapy_plan"], 1):
        st.markdown(f"  {i}. {plan}")

with agent_tabs[3]:
    st.markdown(f"**Alert:** {m_out['alert']}")
    st.markdown(f"**Trend Score:** {m_out['trend']:.3f}")

# ── Score History Chart ────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Score History")
if len(st.session_state.score_history) > 0:
    import pandas as pd

    df = pd.DataFrame(
        {
            "Screening #": range(1, len(st.session_state.score_history) + 1),
            "Score": st.session_state.score_history,
        }
    )
    st.line_chart(df.set_index("Screening #"))

    # Threshold reference lines
    st.caption(
        "Thresholds: < 0.3 = Low Risk | 0.3–0.6 = Monitor | > 0.6 = Clinical Review"
    )

# ── Save results ───────────────────────────────────────────────────────
result = {
    "input": uploaded.name,
    "face_score": f_score,
    "behavior_score": b_score,
    "fused_score": final_score,
    "screening": s_out,
    "clinical": c_out,
    "therapy": t_out,
    "monitoring": m_out,
}
save_result(result)

# ── Footer ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Autism AI Screening Prototype V3 — Multi-Agent Pipeline | "
    "MobileNetV2 + LSTM | For research & competition use only"
)
