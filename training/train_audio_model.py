"""
Train Audio Speech Model (CNN + GRU)
Generates synthetic MFCC/prosody features with ASD-correlated patterns and
trains a 1D-CNN → GRU → Dense binary classifier.

ASD audio markers (from literature):
  - Lower pitch variability / flat prosody
  - Higher pause ratio (longer, more frequent pauses)
  - Lower speech rate
  - Higher jitter (irregular pitch)
  - Atypical MFCC patterns (reduced spectral variation)
"""
import os, sys, json
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

SEED = 42
np.random.seed(SEED)

# ── Generate synthetic training data ───────────────────────────────────
N_SAMPLES = 600
N_FEATURES = 40  # Must match utils/audio_features.py
N_TIMESTEPS = 20  # Temporal windows for CNN+GRU

print("=" * 60)
print("  Audio CNN+GRU Model Training (Synthetic Data)")
print("=" * 60)


def generate_synthetic_audio_data(n_samples=N_SAMPLES):
    """
    Generate synthetic audio feature sequences with ASD-correlated patterns.
    Returns X (n_samples, n_timesteps, n_features), y (n_samples,)
    """
    rng = np.random.RandomState(SEED)
    X = np.zeros((n_samples, N_TIMESTEPS, N_FEATURES), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    
    n_asd = n_samples // 2
    
    for i in range(n_samples):
        is_asd = i < n_asd
        y[i] = 1.0 if is_asd else 0.0
        
        for t in range(N_TIMESTEPS):
            feat = np.zeros(N_FEATURES, dtype=np.float32)
            
            # MFCC features (indices 0-25): 13 means + 13 stds
            if is_asd:
                # ASD: less variable MFCCs, flatter spectral profile
                feat[0:13] = rng.normal(-2.0, 3.0, 13)     # means
                feat[13:26] = rng.normal(1.5, 0.5, 13)      # stds (lower)
            else:
                # TD: more variable MFCCs
                feat[0:13] = rng.normal(0.0, 5.0, 13)       # means
                feat[13:26] = rng.normal(3.0, 1.0, 13)      # stds (higher)
            
            # Prosody features (indices 26-33)
            if is_asd:
                feat[26] = rng.normal(180, 30)    # pitch_mean (lower)
                feat[27] = rng.normal(15, 5)      # pitch_std (lower variability)
                feat[28] = rng.normal(50, 20)     # pitch_range (narrower)
                feat[29] = rng.normal(0.03, 0.01) # energy_mean
                feat[30] = rng.normal(0.01, 0.005)# energy_std
                feat[31] = rng.normal(0.08, 0.02) # zero_crossing_rate
                feat[32] = rng.normal(1500, 300)  # spectral_centroid
                feat[33] = rng.normal(1200, 200)  # spectral_bandwidth
            else:
                feat[26] = rng.normal(220, 40)    # pitch_mean (typical)
                feat[27] = rng.normal(35, 10)     # pitch_std (more variable)
                feat[28] = rng.normal(120, 40)    # pitch_range (wider)
                feat[29] = rng.normal(0.05, 0.015)# energy_mean
                feat[30] = rng.normal(0.02, 0.008)# energy_std
                feat[31] = rng.normal(0.06, 0.015)# zero_crossing_rate
                feat[32] = rng.normal(2000, 400)  # spectral_centroid
                feat[33] = rng.normal(1500, 300)  # spectral_bandwidth
            
            # Vocalization dynamics (indices 34-39)
            if is_asd:
                feat[34] = rng.normal(1.5, 0.5)   # speech_rate (slower)
                feat[35] = rng.normal(0.45, 0.1)   # pause_ratio (more pauses)
                feat[36] = rng.normal(12, 4)        # pause_count (more)
                feat[37] = rng.normal(0.8, 0.3)    # mean_pause_duration (longer)
                feat[38] = rng.normal(0.08, 0.03)  # pitch_variability (lower CV)
                feat[39] = rng.normal(0.04, 0.015) # jitter (higher)
            else:
                feat[34] = rng.normal(3.0, 0.8)    # speech_rate (faster)
                feat[35] = rng.normal(0.25, 0.08)  # pause_ratio (fewer pauses)
                feat[36] = rng.normal(6, 3)         # pause_count (fewer)
                feat[37] = rng.normal(0.3, 0.15)   # mean_pause_duration (shorter)
                feat[38] = rng.normal(0.18, 0.05)  # pitch_variability (higher CV)
                feat[39] = rng.normal(0.015, 0.008) # jitter (lower)
            
            # Add temporal variation
            feat += rng.randn(N_FEATURES) * 0.1 * (t / N_TIMESTEPS)
            X[i, t] = feat
    
    # Shuffle
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


print("\n[1/4] Generating synthetic audio features...")
X, y = generate_synthetic_audio_data()
print(f"  Data shape: X={X.shape}, y={y.shape}")
print(f"  ASD: {int(y.sum())}, TD: {int(len(y) - y.sum())}")

# ── Split ──────────────────────────────────────────────────────────────
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# ── Build CNN + GRU model ──────────────────────────────────────────────
print("\n[2/4] Building CNN + GRU architecture...")

import tensorflow as tf
tf.random.set_seed(SEED)

from tensorflow.keras import layers, Model

inputs = layers.Input(shape=(N_TIMESTEPS, N_FEATURES), name="audio_input")

# 1D CNN for local feature extraction
x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', name='conv1')(inputs)
x = layers.BatchNormalization(name='bn1')(x)
x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu', name='conv2')(x)
x = layers.BatchNormalization(name='bn2')(x)
x = layers.MaxPooling1D(pool_size=2, name='pool1')(x)
x = layers.Dropout(0.3, name='drop1')(x)

# GRU for temporal dynamics
x = layers.GRU(64, return_sequences=True, name='gru1')(x)
x = layers.GRU(32, return_sequences=False, name='gru2')(x)
x = layers.Dropout(0.3, name='drop2')(x)

# Classifier head
x = layers.Dense(64, activation='relu', name='fc1')(x)
x = layers.Dropout(0.2, name='drop3')(x)
outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

model = Model(inputs, outputs, name="AudioCNN_GRU")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── Train ──────────────────────────────────────────────────────────────
print("\n[3/4] Training...")

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
]

history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=30,
    batch_size=32,
    callbacks=callbacks,
    verbose=1,
)

# ── Evaluate ───────────────────────────────────────────────────────────
print("\n[4/4] Evaluating...")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"  Test Accuracy: {acc * 100:.1f}%")
print(f"  Test Loss: {loss:.4f}")

# Predictions for AUC
y_pred = model.predict(X_test, verbose=0).flatten()
from sklearn.metrics import roc_auc_score, classification_report
try:
    auc = roc_auc_score(y_test, y_pred)
    print(f"  AUC: {auc:.4f}")
except:
    auc = 0.0

y_pred_cls = (y_pred > 0.5).astype(int)
print("\n  Classification Report:")
print(classification_report(y_test, y_pred_cls, target_names=["Typical", "ASD"]))

# ── Save ───────────────────────────────────────────────────────────────
SAVE_DIR = os.path.join(PROJECT_DIR, "saved_models")
os.makedirs(SAVE_DIR, exist_ok=True)

model_path = os.path.join(SAVE_DIR, "audio_speech_model.keras")
model.save(model_path)
print(f"\n  ✓ Model saved to {model_path}")

# Save metadata
metadata = {
    "model_type": "CNN_GRU",
    "architecture": "Conv1D(64)→Conv1D(128)→MaxPool→GRU(64)→GRU(32)→Dense(64)→Dense(1)",
    "dataset": "synthetic_audio_features",
    "n_samples": N_SAMPLES,
    "n_features": N_FEATURES,
    "n_timesteps": N_TIMESTEPS,
    "accuracy": round(acc, 4),
    "auc": round(auc, 4),
    "loss": round(loss, 4),
    "seed": SEED,
    "audio_markers": [
        "pitch_variability", "pause_ratio", "speech_rate",
        "jitter", "mfcc_profile", "spectral_features"
    ],
    "note": "Trained on synthetic data with ASD-correlated prosody patterns from literature"
}
meta_path = os.path.join(SAVE_DIR, "audio_speech_metadata.json")
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"  ✓ Metadata saved to {meta_path}")

# Also save a scaler (StandardScaler fit on training data flattened)
from sklearn.preprocessing import StandardScaler
import joblib

scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, N_FEATURES))
scaler_path = os.path.join(SAVE_DIR, "audio_speech_scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"  ✓ Scaler saved to {scaler_path}")

print("\n" + "=" * 60)
print("  Audio Model Training Complete!")
print("=" * 60)
