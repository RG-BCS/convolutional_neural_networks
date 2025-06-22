from tensorflow.keras import layers, models
import tensorflow as tf

def build_alexnet(input_shape=(227, 227, 3), num_classes=6):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        
        layers.Conv2D(96, 11, strides=4, padding='valid', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=3, strides=2),
        
        layers.Conv2D(256, 5, strides=1, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=3, strides=2),
        
        layers.Conv2D(384, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(384, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=3, strides=2),
        
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
