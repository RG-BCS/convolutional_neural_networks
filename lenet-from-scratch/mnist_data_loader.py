import numpy as np
from tensorflow import keras

def reshape28to32(x):
    """Pads 28x28 images to 32x32 with zeros.

    Args:
        x (np.ndarray): Input image batch of shape (batch, 28, 28, 1)

    Returns:
        np.ndarray: Padded image batch of shape (batch, 32, 32, 1)
    """
    return np.pad(x, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant')

def load_mnist_data(flatten=False):
    """Loads and preprocesses the MNIST dataset for LeNet.

    Args:
        flatten (bool): If True, returns unpadded 28x28 images.

    Returns:
        Tuple: (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
    """
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize to [0,1]
    X_train_full = X_train_full.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Split validation set (first 10k)
    X_valid = X_train_full[:10000]
    y_valid = y_train_full[:10000]
    X_train = X_train_full[10000:]
    y_train = y_train_full[10000:]

    # Add channel dimension
    X_train = X_train[..., np.newaxis]
    X_valid = X_valid[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    if not flatten:
        X_train = reshape28to32(X_train)
        X_valid = reshape28to32(X_valid)
        X_test = reshape28to32(X_test)

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
