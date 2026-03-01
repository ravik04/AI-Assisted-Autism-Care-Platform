import tensorflow as tf
from keras.layers import TimeDistributed, GlobalAveragePooling2D
from keras.models import Model


base_cnn = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False)

def build_spatial_encoder():
    inputs = tf.keras.Input(shape=(None, 224, 224, 3))
    x = TimeDistributed(base_cnn)(inputs)
    x = TimeDistributed(GlobalAveragePooling2D())(x)

    return Model(inputs, x)