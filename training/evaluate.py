"""
Evaluate a saved face classifier on the test set and print metrics.
"""
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

DATA_ROOT = r"D:\Autism\AutismData\archive\Autistic Children Facial Image Dataset"
TEST_DIR  = os.path.join(DATA_ROOT, "test")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "saved_models",
                          "face_classifier.keras")

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32


def evaluate():
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Loading test data...")
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="binary", shuffle=False,
    )
    class_names = test_ds.class_names
    print(f"Classes: {class_names}")

    # gather ground truth
    y_true = np.concatenate([y.numpy() for _, y in test_ds]).flatten()
    y_pred_prob = model.predict(test_ds).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    print("\n=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_true, y_pred))

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_true, y_pred,
                                target_names=class_names, digits=4))

    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Test Loss: {loss:.4f}  |  Test Accuracy: {acc:.4f}")


if __name__ == "__main__":
    evaluate()
