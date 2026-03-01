"""
Attention-Based Multimodal Fusion Network
Replaces hand-coded weighted averaging with a learned attention mechanism
that discovers optimal modality weighting from data.

Architecture:
  Per-modality embedding → Multi-Head Self-Attention → Weighted Aggregation → Risk Score
"""
import numpy as np
import os

try:
    import tensorflow as tf
    import keras
    from keras import layers, Model, ops
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# All supported modalities
MODALITY_NAMES = ["face", "behavior", "questionnaire", "eye_tracking", "pose", "audio", "eeg"]
N_MODALITIES = len(MODALITY_NAMES)


@keras.saving.register_keras_serializable(package="autism_fusion")
class ModalityEmbedding(layers.Layer):
    """Project each modality score + confidence into an embedding space."""
    def __init__(self, embed_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense = layers.Dense(embed_dim, activation='relu')
        self.norm = layers.LayerNormalization()
    
    def call(self, x):
        # x: (batch, n_modalities, 2) — [score, confidence]
        return self.norm(self.dense(x))
    
    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config


@keras.saving.register_keras_serializable(package="autism_fusion")
class MaskAndMultiply(layers.Layer):
    """Extract availability mask and zero-out missing modalities."""
    def call(self, inputs):
        x, raw_input = inputs
        mask = raw_input[:, :, 1:2]  # (batch, n_modalities, 1)
        mask_broadcast = ops.cast(mask > 0.5, "float32")
        return x * mask_broadcast, mask_broadcast


@keras.saving.register_keras_serializable(package="autism_fusion")
class CrossModalAttention(layers.Layer):
    """Multi-head self-attention across modalities to learn cross-modal relationships."""
    def __init__(self, embed_dim=32, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads
        )
        self.norm = layers.LayerNormalization()
        self.ffn_dense1 = layers.Dense(embed_dim * 2, activation='relu')
        self.ffn_dense2 = layers.Dense(embed_dim)
        self.norm2 = layers.LayerNormalization()
    
    def call(self, x, mask=None):
        # Self-attention with residual
        attn_output = self.attention(x, x, x, attention_mask=mask)
        x = self.norm(x + attn_output)
        # FFN with residual
        ffn_output = self.ffn_dense2(self.ffn_dense1(x))
        return self.norm2(x + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads})
        return config


@keras.saving.register_keras_serializable(package="autism_fusion")
class WeightedFusionAggregation(layers.Layer):
    """Apply attention weights, mask, normalize, and aggregate."""
    def call(self, inputs):
        x, attn_weights, mask_broadcast = inputs
        attn_weights = attn_weights * mask_broadcast  # Mask missing
        attn_sum = ops.sum(attn_weights, axis=1, keepdims=True) + 1e-8
        attn_weights = attn_weights / attn_sum
        weighted = x * attn_weights
        return ops.sum(weighted, axis=1)  # (batch, embed_dim)


def build_attention_fusion_model(n_modalities=N_MODALITIES, embed_dim=32, num_heads=4):
    """
    Build the attention fusion model.
    
    Input: (batch, n_modalities, 2) — each modality has [score, available_flag]
    Output: (batch, 1) — fused risk score
    Also returns attention weights for explainability.
    """
    if not TF_AVAILABLE:
        return None
    
    inputs = layers.Input(shape=(n_modalities, 2), name="modality_inputs")
    
    # Modality embedding
    x = ModalityEmbedding(embed_dim, name="modality_embed")(inputs)
    
    # Create mask and zero-out missing modalities
    x, mask_broadcast = MaskAndMultiply(name="mask_multiply")([x, inputs])
    
    # Cross-modal attention (2 layers)
    x = CrossModalAttention(embed_dim, num_heads, name="cross_attn_1")(x)
    x = CrossModalAttention(embed_dim, num_heads, name="cross_attn_2")(x)
    
    # Attention-weighted aggregation
    attn_weights = layers.Dense(1, activation='softmax', name="fusion_weights")(x)
    fused = WeightedFusionAggregation(name="weighted_agg")([x, attn_weights, mask_broadcast])
    
    # Classification head
    fused = layers.Dense(32, activation='relu', name="fc1")(fused)
    fused = layers.Dropout(0.2, name="drop1")(fused)
    output = layers.Dense(1, activation='sigmoid', name="risk_output")(fused)
    
    model = Model(inputs, output, name="AttentionFusion")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def prepare_fusion_input(modality_scores: dict) -> np.ndarray:
    """
    Convert a modality_scores dict to model input format.
    Returns array of shape (1, N_MODALITIES, 2) — [score, available_flag]
    """
    x = np.zeros((1, N_MODALITIES, 2), dtype=np.float32)
    for i, name in enumerate(MODALITY_NAMES):
        if name in modality_scores and modality_scores[name] is not None:
            x[0, i, 0] = float(modality_scores[name])
            x[0, i, 1] = 1.0  # available
        else:
            x[0, i, 0] = 0.0
            x[0, i, 1] = 0.0  # missing
    return x


def get_attention_weights(model, modality_scores: dict) -> dict:
    """
    Get per-modality attention weights for explainability.
    Returns dict mapping modality name → weight.
    """
    x = prepare_fusion_input(modality_scores)
    
    # Get the fusion_weights layer output
    weight_model = Model(
        inputs=model.input,
        outputs=model.get_layer("fusion_weights").output
    )
    weights = weight_model.predict(x, verbose=0)[0]  # (n_modalities, 1)
    
    result = {}
    for i, name in enumerate(MODALITY_NAMES):
        if name in modality_scores and modality_scores[name] is not None:
            result[name] = float(weights[i, 0])
    
    # Normalize
    total = sum(result.values()) or 1.0
    result = {k: v / total for k, v in result.items()}
    return result


# ── Fallback: hand-coded weighted fusion ───────────────────────────────
FALLBACK_WEIGHTS = {
    "face": 0.20, "behavior": 0.12, "questionnaire": 0.25,
    "eye_tracking": 0.12, "pose": 0.12, "audio": 0.10, "eeg": 0.09,
}

def fallback_fusion(modality_scores: dict) -> float:
    """Weighted average fusion when learned model is unavailable."""
    total_weight = 0.0
    weighted_sum = 0.0
    for name, score in modality_scores.items():
        if score is not None and name in FALLBACK_WEIGHTS:
            w = FALLBACK_WEIGHTS[name]
            weighted_sum += score * w
            total_weight += w
    return weighted_sum / total_weight if total_weight > 0 else 0.5
