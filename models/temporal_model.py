import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Model

def build_temporal_model(feature_dim):
    inputs = tf.keras.Input(shape=(None, feature_dim))

    x = LSTM(64, return_sequences=False)(inputs)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)

    return Model(inputs, outputs)