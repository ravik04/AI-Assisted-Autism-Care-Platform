"""
Train a MobileNetV2 transfer-learning face classifier.
    Classes: autistic / non_autistic
    Dataset: Autistic Children Facial Image Dataset
"""
import os
import sys
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ── paths ──────────────────────────────────────────────────────────────
DATA_ROOT = r"D:\Autism\AutismData\archive\Autistic Children Facial Image Dataset"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR   = os.path.join(DATA_ROOT, "valid")
TEST_DIR  = os.path.join(DATA_ROOT, "test")

SAVE_DIR  = os.path.join(os.path.dirname(__file__), "..", "saved_models")
MODEL_PATH = os.path.join(SAVE_DIR, "face_classifier.keras")

# ── hyper-params ───────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 1e-4
SEED        = 42


def build_datasets():
    """Load train / val / test image datasets."""
    common = dict(image_size=IMG_SIZE, batch_size=BATCH_SIZE,
                  seed=SEED, label_mode="binary")

    print("Loading training data...")
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR, **common)
    class_names = train_ds.class_names
    print(f"  classes: {class_names}")

    print("Loading validation data...")
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        VAL_DIR, **common)

    print("Loading test data...")
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DIR, **common)

    # performance
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds  = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def build_model():
    """MobileNetV2 + classifier head (frozen backbone)."""
    base = MobileNetV2(weights="imagenet", include_top=False,
                       input_shape=(224, 224, 3))
    base.trainable = False                       # freeze backbone

    inputs  = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train():
    train_ds, val_ds, test_ds, class_names = build_datasets()
    print(f"\nClass mapping: 0={class_names[0]}, 1={class_names[1]}")

    model = build_model()
    model.summary()

    os.makedirs(SAVE_DIR, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
    ]

    print("\n=== TRAINING ===")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    print("\n=== EVALUATION ON TEST SET ===")
    loss, acc = model.evaluate(test_ds)
    print(f"Test Loss: {loss:.4f}  |  Test Accuracy: {acc:.4f}")

    # save final
    model.save(MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")
    return history


if __name__ == "__main__":
    train()
