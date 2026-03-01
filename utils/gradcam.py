"""
Grad-CAM heatmap generation for the MobileNetV2 face classifier.
Highlights which facial regions the model focuses on.
Works with nested models (e.g. MobileNetV2 as a single layer) in TF 2.20+.
"""
import numpy as np
import tensorflow as tf
import cv2


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    """
    Produce a Grad-CAM heatmap for a single image.

    Uses a two-step approach that works with nested sub-models:
      1. Forward pass through the base (frozen) model to get the conv feature map
      2. GradientTape from the feature map through the classifier head

    Parameters
    ----------
    img_array : np.ndarray  – shape (1, 224, 224, 3), raw pixel values
    model     : tf.keras.Model – the full classifier (base + head)

    Returns
    -------
    heatmap : np.ndarray – shape (H, W), values in [0, 1]
    """
    # ── Identify the base (MobileNetV2) sub-model and the head layers ──
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is None:
        raise ValueError("No nested base model found in the classifier")

    # Find last Conv2D inside the base model
    target_layer = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                              tf.keras.layers.DepthwiseConv2D)):
            target_layer = layer
            break

    if target_layer is None:
        raise ValueError("No Conv2D layer found inside the base model")

    # Build a mini-model that outputs both the target conv output and final output
    conv_output = target_layer.output          # e.g. (batch, 7, 7, 1280)
    base_output = base_model.output            # same as conv_output for MobileNetV2

    # Sub-model: input → conv feature map
    feature_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=conv_output,
    )

    # ── Forward pass to get conv features (with GradientTape) ──────────
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        # Get conv features (we need gradient w.r.t. this)
        conv_features = feature_model(img_tensor, training=False)
        tape.watch(conv_features)

        # Continue through the rest of the model
        # base_model output → GAP → Dense head
        # We replicate the forward pass through remaining layers
        x = conv_features
        # Track if we've passed the base model layer
        past_base = False
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                # Replace base model forward with our conv_features
                # We need to apply GAP and the rest
                past_base = True
                continue
            if layer == model.layers[0]:
                # Skip input layer
                continue
            if past_base:
                x = layer(x, training=False)

        predictions = x

        # For binary sigmoid → use the single output
        if predictions.shape[-1] == 1:
            loss = predictions[:, 0]
        else:
            loss = predictions[:, tf.argmax(predictions[0])]

    # ── Compute Grad-CAM ──────────────────────────────────────────────
    grads = tape.gradient(loss, conv_features)
    if grads is None:
        raise RuntimeError("Gradients are None — cannot compute Grad-CAM")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (channels,)

    conv_out = conv_features[0]                            # (H, W, C)
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]     # (H, W, 1)
    heatmap = tf.squeeze(heatmap)                          # (H, W)

    # ReLU + normalise to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay a Grad-CAM heatmap on the original image.

    Parameters
    ----------
    original_img : np.ndarray – (H, W, 3) uint8, original image (BGR or RGB)
    heatmap      : np.ndarray – (h, w) float in [0, 1]
    alpha        : float      – blending factor

    Returns
    -------
    superimposed : np.ndarray – (H, W, 3) uint8 RGB
    """
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    jet = cv2.applyColorMap(heatmap_uint8, colormap)
    jet_rgb = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)

    # Ensure original is RGB uint8
    if original_img.dtype != np.uint8:
        original_img = np.uint8(np.clip(original_img, 0, 255))

    superimposed = cv2.addWeighted(original_img, 1 - alpha, jet_rgb, alpha, 0)
    return superimposed
