# cnn_models/cnn_model.py

from tensorflow.keras import layers, models, regularizers


def build_cnn_model(input_shape: tuple, num_classes: int):
    """
    Build and compile a CNN for mel spectrogram classification.

    Args:
        input_shape : Shape of one input sample, e.g. (128, 128, 1).
        num_classes : Number of output classes (4 for ICBHI).

    Returns:
        Compiled Keras Sequential model.
    """
    l2 = regularizers.l2(1e-4)

    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Block 1 
        # Small 3×3 kernels detect local frequency-time patterns (e.g. short
        # crackle transients). 32 filters keeps early layers lightweight.
        layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # Block 2 
        # 64 filters capture mid-level patterns such as wheeze harmonics
        # spread across frequency bins.
        layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # Block 3 
        # 128 filters combine lower-level features into high-level
        # representations of cycle type.
        layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # Block 4 
        # Extra conv block increases model capacity without adding a huge
        # number of parameters (no pooling here to preserve spatial info).
        layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.GlobalAveragePooling2D(),  # replaces Flatten+Dense bottleneck;
                                          # averages each feature map → more
                                          # parameter-efficient and regularising

        # Classifier head 
        layers.Dense(256, kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.5),

        layers.Dense(128, kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation="softmax"),
    ], name="cnn_mel_spectrogram")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model
