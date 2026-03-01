"""
Autism AI Prototype V3 — Multi-Agent Pipeline
Uses trained MobileNetV2 face classifier + LSTM behavior model.
Falls back to pseudo-scoring if trained models are not available.
"""
import os
import sys
import numpy as np
import tensorflow as tf

from agents.screening_agent import screening_agent
from agents.clinical_agent import clinical_agent
from agents.therapy_agent import therapy_agent
from agents.monitoring_agent import monitoring_agent
from utils.logger import save_result

# ── paths ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
FACE_MODEL = os.path.join(BASE_DIR, "saved_models", "face_classifier.keras")
BEH_MODEL  = os.path.join(BASE_DIR, "saved_models", "behavior_lstm.keras")

IMG_SIZE = (224, 224)

# ── load trained models ────────────────────────────────────────────────
face_model    = None
behavior_model = None

if os.path.exists(FACE_MODEL):
    print("Loading trained face classifier...")
    face_model = tf.keras.models.load_model(FACE_MODEL)
    print("  ✓ Face classifier loaded")
else:
    print("  ⚠ Face classifier not found — will use fallback scoring")

if os.path.exists(BEH_MODEL):
    print("Loading trained behavior LSTM...")
    behavior_model = tf.keras.models.load_model(BEH_MODEL)
    print("  ✓ Behavior LSTM loaded")
else:
    print("  ⚠ Behavior LSTM not found — will use fallback scoring")

score_memory = []


# ── helpers ────────────────────────────────────────────────────────────
def load_image(path):
    """Load a single image for the face classifier."""
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


def extract_frames_from_video(video_path, max_frames=16):
    """Pull evenly-spaced frames from a video file."""
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
            frame = cv2.resize(frame, IMG_SIZE)
            frames.append(frame.astype(np.float32))
    cap.release()
    return np.array(frames)  # (N, 224, 224, 3)


# ── scoring functions ──────────────────────────────────────────────────
def score_face(image_path):
    """Run face classifier on a single image. Returns P(autistic)."""
    if face_model is None:
        return 0.5  # fallback
    img = load_image(image_path)
    pred = face_model.predict(img, verbose=0)[0][0]
    # class 0 = autistic, class 1 = non_autistic → P(autistic) = 1 - pred
    return float(1.0 - pred)


def score_face_from_frames(frames):
    """Average face-classifier score across multiple video frames."""
    if face_model is None or len(frames) == 0:
        return 0.5
    preds = face_model.predict(frames, verbose=0).flatten()
    return float(1.0 - np.mean(preds))


def score_behavior(video_path):
    """Run behavior LSTM on optical-flow–like features from video frames."""
    if behavior_model is None:
        return 0.5  # fallback
    frames = extract_frames_from_video(video_path, max_frames=16)
    if len(frames) == 0:
        return 0.5

    # Extract MobileNetV2 features per frame
    extractor = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False,
        input_shape=(224, 224, 3))
    gap = tf.keras.layers.GlobalAveragePooling2D()

    feats = []
    for f in frames:
        inp = tf.keras.applications.mobilenet_v2.preprocess_input(
            np.expand_dims(f, 0))
        feat = gap(extractor(inp, training=False))
        feats.append(feat.numpy().flatten())

    feat_seq = np.array(feats)       # (16, 1280)
    feat_seq = np.expand_dims(feat_seq, 0)  # (1, 16, 1280)
    pred = behavior_model.predict(feat_seq, verbose=0)[0][0]
    return float(pred)


def fuse_scores(face_score, behavior_score):
    """Weighted fusion: face 60%, behavior 40%."""
    fused = 0.6 * face_score + 0.4 * behavior_score
    return float(np.clip(fused, 0, 1))


# ── main pipeline ─────────────────────────────────────────────────────
def run_pipeline(input_path):
    """
    Accepts an image path OR a video path.
    - Image  → face score only
    - Video  → face + behavior fusion
    """
    ext = os.path.splitext(input_path)[1].lower()
    is_video = ext in (".mp4", ".avi", ".mov", ".mkv", ".webm")

    if is_video:
        print(f"\n[INPUT] Video: {input_path}")
        frames = extract_frames_from_video(input_path)
        f_score = score_face_from_frames(frames)
        b_score = score_behavior(input_path)
        score   = fuse_scores(f_score, b_score)
        print(f"  Face score:     {f_score:.3f}")
        print(f"  Behavior score: {b_score:.3f}")
        print(f"  Fused score:    {score:.3f}")
    else:
        print(f"\n[INPUT] Image: {input_path}")
        f_score = score_face(input_path)
        b_score = None
        score   = f_score
        print(f"  Face score: {score:.3f}")

    score_memory.append(score)

    # multi-agent chain
    s_out = screening_agent(score)
    c_out = clinical_agent(s_out)
    t_out = therapy_agent(c_out)
    m_out = monitoring_agent(score_memory)

    print("\n=== SCREENING ===", s_out)
    print("\n=== CLINICAL ===", c_out)
    print("\n=== THERAPY ===", t_out)
    print("\n=== MONITORING ===", m_out)
    print("\nScore History:", [round(s, 3) for s in score_memory])

    result = {
        "input": input_path,
        "face_score": f_score,
        "behavior_score": b_score,
        "fused_score": score,
        "screening": s_out,
        "clinical": c_out,
        "therapy": t_out,
        "monitoring": m_out,
    }
    save_result(result)
    print("\nResults saved → output.json")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # default demo: grab a random test image
        test_img = os.path.join(
            r"D:\Autism\AutismData\archive",
            r"Autistic Children Facial Image Dataset\test\autistic",
            "001.jpg")
        if os.path.exists(test_img):
            run_pipeline(test_img)
        else:
            print("Usage: python app.py <image_or_video_path>")
    else:
        run_pipeline(sys.argv[1])