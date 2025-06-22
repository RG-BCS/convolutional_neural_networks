import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_lenet(input_shape=(32, 32, 1), num_classes=10):
    """Builds the LeNet-5 model architecture.

    Args:
        input_shape (tuple): Shape of the input images (must be 32x32x1).
        num_classes (int): Number of output classes.

    Returns:
        keras.Model: Compiled LeNet-5 model.
    """
    model = keras.Sequential([
        layers.InputLayer(input_shape=input_shape),

        layers.Conv2D(filters=6, kernel_size=5, activation='tanh', padding='valid'),
        layers.AveragePooling2D(pool_size=2),
        layers.Activation('tanh'),

        layers.Conv2D(filters=16, kernel_size=5, activation='tanh', padding='valid'),
        layers.AveragePooling2D(pool_size=2),
        layers.Activation('tanh'),

        layers.Conv2D(filters=120, kernel_size=5, activation='tanh', padding='valid'),
        layers.Flatten(),
        layers.Dense(84, activation='tanh'),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model
