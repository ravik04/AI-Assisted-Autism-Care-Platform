"""
Dataset loader for facial image classification.
Loads images from directory structure: class_folder/images.jpg
"""
import tensorflow as tf
import os

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def load_image_dataset(data_dir, subset=None, validation_split=None):
    """
    Load image dataset from directory.
    Expects structure: data_dir/class_a/*.jpg, data_dir/class_b/*.jpg
    """
    kwargs = dict(
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        label_mode="binary",
    )
    if validation_split is not None and subset is not None:
        kwargs["validation_split"] = validation_split
        kwargs["subset"] = subset

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir, **kwargs
    )
    class_names = dataset.class_names
    print(f"  Classes: {class_names}  |  Batches: {tf.data.experimental.cardinality(dataset).numpy()}")

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset, class_names


def load_train_val_test(train_dir, val_dir, test_dir):
    """Load pre-split train / val / test directories."""
    print("[TRAIN]")
    train_ds, class_names = load_image_dataset(train_dir)
    print("[VAL]")
    val_ds, _ = load_image_dataset(val_dir)
    print("[TEST]")
    test_ds, _ = load_image_dataset(test_dir)
    return train_ds, val_ds, test_ds, class_names
