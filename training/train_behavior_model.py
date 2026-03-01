"""
Train an LSTM behavior classifier on optical-flow sequences.
Labels derived from folder suffix:  _y = correct,  _n = incorrect,  _i = incomplete
We map:  y → 0 (typical),  n/i → 1 (atypical)   — binary screening framing.
Each clip folder has ~100-200 optical flow x/y frame pairs.
We sample SEQUENCE_LENGTH frames, resize, stack x+y into 2-channel images,
run through MobileNetV2 feature extractor, then feed into LSTM.
"""
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (LSTM, Dense, Dropout,
                                     GlobalAveragePooling2D, TimeDistributed)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ── paths ──────────────────────────────────────────────────────────────
FLOW_ROOT  = r"D:\Autism\AutismData\optical_flow-001\optical_flow_jpg"
SAVE_DIR   = os.path.join(os.path.dirname(__file__), "..", "saved_models")
MODEL_PATH = os.path.join(SAVE_DIR, "behavior_lstm.keras")

# ── hyper-params ───────────────────────────────────────────────────────
IMG_SIZE         = (112, 112)   # smaller for speed  (optical flow doesn't need 224)
SEQUENCE_LENGTH  = 16           # frames per clip
BATCH_SIZE       = 8
EPOCHS           = 10
LR               = 1e-4
SEED             = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ── data helpers ───────────────────────────────────────────────────────

def _label_from_folder(name):
    """y → 0 (typical), n/i → 1 (atypical)."""
    suffix = name.rsplit("_", 1)[-1]
    return 0 if suffix == "y" else 1


def _load_flow_frame(x_path, y_path, size):
    """Load optical-flow x/y pair, resize, normalise, stack into 3-ch."""
    x_img = tf.keras.utils.load_img(x_path, color_mode="grayscale",
                                     target_size=size)
    y_img = tf.keras.utils.load_img(y_path, color_mode="grayscale",
                                     target_size=size)
    x_arr = np.array(x_img, dtype=np.float32) / 255.0
    y_arr = np.array(y_img, dtype=np.float32) / 255.0
    # stack into 3-channel (x, y, magnitude) so MobileNetV2 input works
    mag = np.sqrt(x_arr ** 2 + y_arr ** 2)
    return np.stack([x_arr, y_arr, mag], axis=-1)


def _load_clip(clip_dir, seq_len, size):
    """Sample seq_len frames from a clip directory."""
    x_frames = sorted([f for f in os.listdir(clip_dir) if f.endswith("_x.jpg")])
    if len(x_frames) == 0:
        return None

    # sample evenly
    indices = np.linspace(0, len(x_frames) - 1, seq_len, dtype=int)
    frames = []
    for idx in indices:
        x_name = x_frames[idx]
        y_name = x_name.replace("_x.jpg", "_y.jpg")
        x_path = os.path.join(clip_dir, x_name)
        y_path = os.path.join(clip_dir, y_name)
        if not os.path.exists(y_path):
            return None
        frames.append(_load_flow_frame(x_path, y_path, size))

    return np.array(frames)   # (seq_len, H, W, 3)


def build_dataset():
    """Walk all activity folders, load clips, return X, y arrays."""
    X_all, y_all = [], []
    root = FLOW_ROOT

    activity_dirs = sorted([d for d in os.listdir(root)
                            if os.path.isdir(os.path.join(root, d))])

    for act in activity_dirs:
        act_path = os.path.join(root, act)
        clips = sorted([c for c in os.listdir(act_path)
                        if os.path.isdir(os.path.join(act_path, c))])
        for clip_name in clips:
            clip_path = os.path.join(act_path, clip_name)
            label = _label_from_folder(clip_name)
            seq = _load_clip(clip_path, SEQUENCE_LENGTH, IMG_SIZE)
            if seq is not None:
                X_all.append(seq)
                y_all.append(label)

    X_all = np.array(X_all)   # (N, seq, H, W, 3)
    y_all = np.array(y_all)   # (N,)
    print(f"Loaded {len(X_all)} clips  |  typical(y)={np.sum(y_all==0)}  atypical(n/i)={np.sum(y_all==1)}")
    return X_all, y_all


def build_feature_extractor():
    """Frozen MobileNetV2 → GAP → 1280-d feature per frame."""
    base = MobileNetV2(weights="imagenet", include_top=False,
                       input_shape=(*IMG_SIZE, 3))
    base.trainable = False
    inp = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    x = base(x, training=False)
    x = GlobalAveragePooling2D()(x)
    return Model(inp, x, name="flow_feature_extractor")


def extract_features(extractor, X):
    """Extract spatial features for every frame in every clip.
    X shape: (N, seq, H, W, 3) → (N, seq, 1280)
    """
    N, S = X.shape[0], X.shape[1]
    flat = X.reshape(N * S, *X.shape[2:])          # (N*S, H, W, 3)
    feats = extractor.predict(flat, batch_size=32, verbose=1)  # (N*S, 1280)
    return feats.reshape(N, S, -1)                  # (N, seq, 1280)


def build_lstm_model(feature_dim=1280):
    """LSTM classifier on top of pre-extracted features."""
    inp = tf.keras.Input(shape=(SEQUENCE_LENGTH, feature_dim))
    x = LSTM(64, return_sequences=False)(inp)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def train():
    # 1) load clips
    X, y = build_dataset()

    # 2) extract spatial features with frozen MobileNetV2
    print("\nExtracting MobileNetV2 features...")
    extractor = build_feature_extractor()
    X_feat = extract_features(extractor, X)   # (N, seq, 1280)

    # 3) train/val split (80/20 stratified)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_feat, y, test_size=0.2, random_state=SEED, stratify=y)

    print(f"Train: {len(X_train)}  |  Val: {len(X_val)}")

    # 4) build & train LSTM
    model = build_lstm_model(X_feat.shape[-1])
    model.summary()

    os.makedirs(SAVE_DIR, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
    ]

    print("\n=== TRAINING LSTM ===")
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE,
              callbacks=callbacks)

    print("\n=== EVALUATION ===")
    loss, acc = model.evaluate(X_val, y_val)
    print(f"Val Loss: {loss:.4f}  |  Val Accuracy: {acc:.4f}")

    model.save(MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")


if __name__ == "__main__":
    train()
