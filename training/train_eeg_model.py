"""
Train EEG Neural Signal Model (1D-CNN Classifier)
Generates synthetic EEG band-power features with ASD-correlated patterns
and trains a 1D-CNN classifier.

ASD EEG markers (from literature):
  - Elevated theta/beta ratio (poor attention regulation)
  - Reduced alpha asymmetry (atypical lateralization)
  - Higher delta & theta power (cortical hypoactivation)
  - Reduced alpha power (poor sensory gating)
  - Altered spectral entropy
"""
import os, sys, json
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

SEED = 42
np.random.seed(SEED)

N_SAMPLES = 500
N_FEATURES = 20  # Must match utils/eeg_features.py

print("=" * 60)
print("  EEG Neural Signal Model Training (Synthetic Data)")
print("=" * 60)


def generate_synthetic_eeg_data(n_samples=N_SAMPLES):
    """
    Generate synthetic EEG features with ASD-correlated patterns.
    Feature order matches eeg_features.py FEATURE_NAMES.
    """
    rng = np.random.RandomState(SEED)
    X = np.zeros((n_samples, N_FEATURES), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    
    n_asd = n_samples // 2
    
    for i in range(n_samples):
        is_asd = i < n_asd
        y[i] = 1.0 if is_asd else 0.0
        
        if is_asd:
            # ASD: elevated theta, reduced alpha, high theta/beta ratio
            delta = rng.normal(0.28, 0.05)
            theta = rng.normal(0.32, 0.05)
            alpha = rng.normal(0.15, 0.04)
            beta  = rng.normal(0.16, 0.03)
            gamma = rng.normal(0.09, 0.02)
            theta_beta = theta / (beta + 1e-6)      # ~2.0 (elevated)
            theta_alpha = theta / (alpha + 1e-6)     # ~2.1
            alpha_beta = alpha / (beta + 1e-6)
            delta_theta = delta / (theta + 1e-6)
            alpha_asym = rng.normal(-0.08, 0.04)     # reduced asymmetry
            beta_asym = rng.normal(0.01, 0.03)
            theta_asym = rng.normal(-0.02, 0.03)
            frontal_asym = rng.normal(-0.06, 0.04)
            parietal_asym = rng.normal(-0.04, 0.03)
            spec_entropy = rng.normal(2.8, 0.4)      # lower entropy
            sample_ent = rng.normal(1.2, 0.3)
            hjorth_act = rng.normal(1.2, 0.3)        # higher
            hjorth_mob = rng.normal(0.5, 0.1)
            corr_mean = rng.normal(0.35, 0.08)       # lower connectivity
            coh_alpha = rng.normal(0.30, 0.08)
        else:
            # Typical: balanced band powers
            delta = rng.normal(0.22, 0.05)
            theta = rng.normal(0.20, 0.04)
            alpha = rng.normal(0.28, 0.05)
            beta  = rng.normal(0.20, 0.04)
            gamma = rng.normal(0.10, 0.02)
            theta_beta = theta / (beta + 1e-6)       # ~1.0 (typical)
            theta_alpha = theta / (alpha + 1e-6)
            alpha_beta = alpha / (beta + 1e-6)
            delta_theta = delta / (theta + 1e-6)
            alpha_asym = rng.normal(0.05, 0.04)      # normal asymmetry
            beta_asym = rng.normal(0.03, 0.03)
            theta_asym = rng.normal(0.02, 0.03)
            frontal_asym = rng.normal(0.06, 0.04)
            parietal_asym = rng.normal(0.04, 0.03)
            spec_entropy = rng.normal(3.5, 0.4)      # higher entropy
            sample_ent = rng.normal(1.8, 0.3)
            hjorth_act = rng.normal(0.6, 0.2)        # lower
            hjorth_mob = rng.normal(0.35, 0.08)
            corr_mean = rng.normal(0.55, 0.1)        # higher connectivity
            coh_alpha = rng.normal(0.50, 0.1)
        
        # Normalize band powers
        total = delta + theta + alpha + beta + gamma
        X[i] = [
            delta/total, theta/total, alpha/total, beta/total, gamma/total,
            theta_beta, theta_alpha, alpha_beta, delta_theta,
            alpha_asym, beta_asym, theta_asym, frontal_asym, parietal_asym,
            spec_entropy, sample_ent, hjorth_act, hjorth_mob,
            corr_mean, coh_alpha,
        ]
    
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


print("\n[1/4] Generating synthetic EEG features...")
X, y = generate_synthetic_eeg_data()
print(f"  Data shape: X={X.shape}, y={y.shape}")
print(f"  ASD: {int(y.sum())}, TD: {int(len(y) - y.sum())}")

# ── Split ──────────────────────────────────────────────────────────────
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Reshape for 1D-CNN: (samples, features, 1)
X_train_cnn = X_train_sc.reshape(-1, N_FEATURES, 1)
X_test_cnn = X_test_sc.reshape(-1, N_FEATURES, 1)

print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# ── Build 1D-CNN model ────────────────────────────────────────────────
print("\n[2/4] Building 1D-CNN architecture...")

import tensorflow as tf
tf.random.set_seed(SEED)
from tensorflow.keras import layers, Model

inputs = layers.Input(shape=(N_FEATURES, 1), name="eeg_input")
x = layers.Conv1D(32, 3, padding='same', activation='relu', name='conv1')(inputs)
x = layers.BatchNormalization(name='bn1')(x)
x = layers.Conv1D(64, 3, padding='same', activation='relu', name='conv2')(x)
x = layers.BatchNormalization(name='bn2')(x)
x = layers.GlobalAveragePooling1D(name='gap')(x)
x = layers.Dense(32, activation='relu', name='fc1')(x)
x = layers.Dropout(0.3, name='drop1')(x)
outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

model = Model(inputs, outputs, name="EEG_CNN")
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── Train ──────────────────────────────────────────────────────────────
print("\n[3/4] Training...")
history = model.fit(
    X_train_cnn, y_train,
    validation_split=0.15,
    epochs=30,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
    ],
    verbose=1,
)

# ── Evaluate ───────────────────────────────────────────────────────────
print("\n[4/4] Evaluating...")
loss, acc = model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"  Test Accuracy: {acc * 100:.1f}%")

y_pred = model.predict(X_test_cnn, verbose=0).flatten()
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

model_path = os.path.join(SAVE_DIR, "eeg_neural_model.keras")
model.save(model_path)
print(f"\n  ✓ Model saved to {model_path}")

import joblib
scaler_path = os.path.join(SAVE_DIR, "eeg_neural_scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"  ✓ Scaler saved to {scaler_path}")

metadata = {
    "model_type": "1D_CNN",
    "architecture": "Conv1D(32)→Conv1D(64)→GAP→Dense(32)→Dense(1)",
    "dataset": "synthetic_eeg_bandpower",
    "n_samples": N_SAMPLES,
    "n_features": N_FEATURES,
    "accuracy": round(acc, 4),
    "auc": round(auc, 4),
    "eeg_markers": [
        "theta_beta_ratio", "alpha_asymmetry", "spectral_entropy",
        "band_power_distribution", "hjorth_parameters", "coherence"
    ],
    "note": "Trained on synthetic data with ASD-correlated EEG patterns from literature"
}
meta_path = os.path.join(SAVE_DIR, "eeg_neural_metadata.json")
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"  ✓ Metadata saved to {meta_path}")

print("\n" + "=" * 60)
print("  EEG Model Training Complete!")
print("=" * 60)
