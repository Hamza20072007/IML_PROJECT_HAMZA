"""Train a 1D CNN using the project's preprocessing and validation split."""

from pathlib import Path
import sys

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_validation_split.train_validation_split import create_train_validation_split


TIME_STEPS = 250
CHANNELS = 23


def prepare_cnn_data():
    """Split, scale, reshape, and encode data without validation leakage."""
    X_train, X_val, y_train, y_val, _, _ = create_train_validation_split()

    expected_feature_count = TIME_STEPS * CHANNELS
    if X_train.shape[1] != expected_feature_count:
        raise ValueError(
            f"Expected {expected_feature_count} features, got {X_train.shape[1]}"
        )

    # Fit only on training values. Validation data uses the same statistics.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Conv1D expects (samples, time steps, channels).
    X_train_cnn = X_train_scaled.reshape(-1, TIME_STEPS, CHANNELS)
    X_val_cnn = X_val_scaled.reshape(-1, TIME_STEPS, CHANNELS)

    # Fit label mapping on training labels, then use it unchanged for validation.
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_val_encoded = encoder.transform(y_val)
    num_classes = len(encoder.classes_)

    y_train_one_hot = np.eye(num_classes, dtype=np.float32)[y_train_encoded]
    y_val_one_hot = np.eye(num_classes, dtype=np.float32)[y_val_encoded]

    return (
        X_train_cnn,
        X_val_cnn,
        y_train_one_hot,
        y_val_one_hot,
        encoder,
        scaler,
    )


def build_1d_cnn(input_shape, num_classes):
    """Build and compile the 1D CNN model."""
    try:
        from tensorflow.keras.layers import (
            Conv1D,
            Dense,
            GlobalAveragePooling1D,
            Input,
            MaxPooling1D,
        )
        from tensorflow.keras.models import Sequential
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "TensorFlow is required to train the CNN. Install it in a compatible "
            "Python environment before running this script."
        ) from error

    model = Sequential(
        [
            Input(shape=input_shape),
            Conv1D(filters=64, kernel_size=5, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            GlobalAveragePooling1D(),
            Dense(num_classes, activation='softmax'),
        ]
    )
    model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']
    )
    return model


def train_cnn(epochs=20, batch_size=32):
    """Prepare the data, train the CNN, and return model, history, scaler, encoder."""
    X_train, X_val, y_train, y_val, encoder, scaler = prepare_cnn_data()
    model = build_1d_cnn(X_train.shape[1:], len(encoder.classes_))
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )
    return model, history, scaler, encoder


if __name__ == '__main__':
    X_train, X_val, y_train, y_val, encoder, _ = prepare_cnn_data()
    print(f"Training input shape: {X_train.shape}")
    print(f"Validation input shape: {X_val.shape}")
    print(f"Classes: {list(encoder.classes_)}")

    model = build_1d_cnn(X_train.shape[1:], len(encoder.classes_))
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)
