import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils import get_image_generators,plot_predictions, plot_confusion_matrix
from models import build_alexnet

# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Paths to directories
test_dir = '/kaggle/input/intel-image-classification/seg_test/seg_test'
train_dir = '/kaggle/input/intel-image-classification/seg_train/seg_train'
class_names_dict = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}

# Image size and batch size
img_size = (227, 227)
batch_size = 32
num_epochs = 30

# Get image data generators
train_gen, val_gen, test_gen = get_image_generators(train_dir, test_dir, img_size, batch_size)

# Build the AlexNet model
alexnet_model = build_alexnet(input_shape=(227, 227, 3), num_classes=len(class_names_dict))

# Callbacks for early stopping and learning rate reduction
early_stop = EarlyStopping(patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
callbacks = [early_stop, reduce_lr]

# Compile the model
alexnet_model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

# Evaluate model before training
X_test, y_test = next(test_gen)
_, acc = alexnet_model.evaluate(X_test, y_test, verbose=0)
print(f"AlexNet Model Test Accuracy Before Training: {acc * 100:.2f}")

# Train the model
history = alexnet_model.fit(train_gen, epochs=num_epochs, validation_data=val_gen, callbacks=callbacks)

# Evaluate model after training
_, acc = alexnet_model.evaluate(X_test, y_test, verbose=0)
print(f"AlexNet Model Test Accuracy After Training: {acc * 100:.2f}")

# Plot accuracy and loss curves
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# Plot confusion matrix and predictions
plot_confusion_matrix(alexnet_model, X_test, y_test, class_names_dict)
plot_predictions(alexnet_model, X_test, y_test, class_names_dict, row=4, col=8, figsize=(15, 3))
