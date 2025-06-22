# models/models.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_mlp_model(input_shape, num_classes):
    """Builds a simple MLP model for image classification."""
    model = keras.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Flatten(),
        layers.Dense(300, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.SGD(),
        metrics=['accuracy']
    )
    
    return model

def build_cnn_model(input_shape, num_classes):
    """Builds a CNN model for image classification."""
    model = keras.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(64, 7, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    
    return model
