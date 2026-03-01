"""
Train Attention Fusion Model
Generates training data by combining simulated multi-modal outputs
and trains the attention-based fusion network.
"""
import os, sys, json
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

SEED = 42
np.random.seed(SEED)

print("=" * 60)
print("  Attention Fusion Model Training")
print("=" * 60)

from models.attention_fusion import (
    build_attention_fusion_model, MODALITY_NAMES, N_MODALITIES
)

# ── Generate training data ─────────────────────────────────────────────
N_SAMPLES = 1000

def generate_fusion_data(n=N_SAMPLES):
    """
    Simulate multi-modal scores with realistic missing-modality patterns.
    Label: weighted consensus of available modality scores (ground truth approximation).
    """
    rng = np.random.RandomState(SEED)
    X = np.zeros((n, N_MODALITIES, 2), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    
    for i in range(n):
        is_asd = rng.random() < 0.5
        base_risk = rng.beta(5, 2) if is_asd else rng.beta(2, 5)
        
        # Randomly make 1-3 modalities unavailable
        n_available = rng.randint(3, N_MODALITIES + 1)
        available = rng.choice(N_MODALITIES, n_available, replace=False)
        
        for j in range(N_MODALITIES):
            if j in available:
                # Score with modality-specific noise
                noise = rng.normal(0, 0.1)
                score = np.clip(base_risk + noise, 0, 1)
                X[i, j, 0] = score
                X[i, j, 1] = 1.0  # available
            else:
                X[i, j, 0] = 0.0
                X[i, j, 1] = 0.0  # missing
        
        # Ground truth: whether the case is actually ASD
        y[i] = 1.0 if is_asd else 0.0
    
    idx = rng.permutation(n)
    return X[idx], y[idx]

print("\n[1/3] Generating multi-modal fusion training data...")
X, y = generate_fusion_data()
print(f"  Shape: X={X.shape}, y={y.shape}")

split = int(0.8 * N_SAMPLES)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ── Build & Train ──────────────────────────────────────────────────────
print("\n[2/3] Building Attention Fusion model...")
model = build_attention_fusion_model()
model.summary()

print("\n[3/3] Training...")
import tensorflow as tf
tf.random.set_seed(SEED)

history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=30,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    ],
    verbose=1,
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n  Test Accuracy: {acc * 100:.1f}%")

from sklearn.metrics import roc_auc_score
y_pred = model.predict(X_test, verbose=0).flatten()
try:
    auc = roc_auc_score(y_test, y_pred)
    print(f"  AUC: {auc:.4f}")
except:
    auc = 0.0

# ── Save ───────────────────────────────────────────────────────────────
SAVE_DIR = os.path.join(PROJECT_DIR, "saved_models")
model_path = os.path.join(SAVE_DIR, "attention_fusion.keras")
model.save(model_path)
print(f"\n  ✓ Model saved to {model_path}")

metadata = {
    "model_type": "AttentionFusion",
    "architecture": "ModalityEmbed→2×CrossModalAttention(4heads)→WeightedAgg→Dense(32)→Dense(1)",
    "modalities": MODALITY_NAMES,
    "n_modalities": N_MODALITIES,
    "accuracy": round(acc, 4),
    "auc": round(auc, 4),
    "n_samples": N_SAMPLES,
    "handles_missing_modalities": True,
}
with open(os.path.join(SAVE_DIR, "attention_fusion_metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 60)
print("  Attention Fusion Training Complete!")
print("=" * 60)
