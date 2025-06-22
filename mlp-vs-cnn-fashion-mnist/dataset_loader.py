# data/dataset_loader.py

import numpy as np
import tensorflow as tf
from tensorflow import keras

def load_fashion_mnist(seed=42, add_channel_dim=True):
    """Loads and preprocesses the Fashion MNIST dataset."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    fashion = keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion.load_data()

    # Normalize pixel values
    X_train_full = X_train_full / 255.0
    X_test = X_test / 255.0

    # Split training into training + validation
    X_valid = X_train_full[:10000]
    y_valid = y_train_full[:10000]
    X_train = X_train_full[10000:]
    y_train = y_train_full[10000:]

    # Add channel dimension if needed (for CNN)
    if add_channel_dim:
        X_train = X_train[..., np.newaxis]
        X_valid = X_valid[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test), class_names
