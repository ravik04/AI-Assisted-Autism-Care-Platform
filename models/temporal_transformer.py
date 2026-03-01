"""
Temporal Fusion Transformer for Video Behavior Analysis
Processes frame-level MobileNetV2 features through positional encoding
and self-attention blocks to model temporal behavioral patterns.

Architecture:
  Frame Features → Positional Encoding → 2× TransformerBlock → Temporal Pool → Dense → Sigmoid
"""
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding for temporal sequences."""
    def __init__(self, max_len=64, embed_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        pe = np.zeros((self.max_len, self.embed_dim))
        position = np.arange(0, self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embed_dim, 2) * -(np.log(10000.0) / self.embed_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
        super().build(input_shape)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]


class TransformerBlock(layers.Layer):
    """Single transformer encoder block with multi-head self-attention and FFN."""
    def __init__(self, embed_dim=128, num_heads=4, ff_dim=256, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
    
    def call(self, x, training=None):
        # Self-attention with residual
        attn_out = self.attention(x, x, x, training=training)
        attn_out = self.dropout1(attn_out, training=training)
        x = self.norm1(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x, training=training)
        ffn_out = self.dropout2(ffn_out, training=training)
        return self.norm2(x + ffn_out)


def build_temporal_transformer(
    sequence_length=16,
    feature_dim=1280,  # MobileNetV2 output dim
    embed_dim=128,
    num_heads=4,
    num_layers=2,
    ff_dim=256,
    dropout=0.1,
):
    """
    Build a Temporal Fusion Transformer for video behavior classification.
    
    Input: (batch, sequence_length, feature_dim) — frame-level CNN features
    Output: (batch, 1) — ASD risk probability
    """
    if not TF_AVAILABLE:
        return None
    
    inputs = layers.Input(shape=(sequence_length, feature_dim), name="frame_features")
    
    # Project to embedding dimension
    x = layers.Dense(embed_dim, name="feature_projection")(inputs)
    x = layers.Dropout(dropout, name="proj_dropout")(x)
    
    # Positional encoding
    x = PositionalEncoding(max_len=sequence_length, embed_dim=embed_dim, name="pos_encoding")(x)
    
    # Transformer blocks
    for i in range(num_layers):
        x = TransformerBlock(
            embed_dim=embed_dim, num_heads=num_heads,
            ff_dim=ff_dim, dropout=dropout, name=f"transformer_{i}"
        )(x)
    
    # Temporal pooling: combine attention-weighted average and max
    avg_pool = layers.GlobalAveragePooling1D(name="avg_pool")(x)
    max_pool = layers.GlobalMaxPooling1D(name="max_pool")(x)
    x = layers.Concatenate(name="pool_concat")([avg_pool, max_pool])
    
    # Classification head
    x = layers.Dense(64, activation='relu', name="fc1")(x)
    x = layers.Dropout(dropout, name="fc_dropout")(x)
    x = layers.Dense(32, activation='relu', name="fc2")(x)
    output = layers.Dense(1, activation='sigmoid', name="output")(x)
    
    model = Model(inputs, output, name="TemporalFusionTransformer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_small_temporal_transformer(sequence_length=16, feature_dim=40):
    """
    Smaller variant for non-CNN features (e.g., audio temporal windows).
    Input: (batch, sequence_length, feature_dim)
    """
    return build_temporal_transformer(
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        embed_dim=64,
        num_heads=2,
        num_layers=2,
        ff_dim=128,
        dropout=0.15,
    )
