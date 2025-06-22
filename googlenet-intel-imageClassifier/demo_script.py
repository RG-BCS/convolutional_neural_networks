# demo_script.py

from model import build_inception_model
from dataset_generator import get_data_generators
from utils import multi_output_generator, plot_confusion_matrix, plot_predictions

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

# === CONFIGURATION ===
train_dir = '/kaggle/input/intel-image-classification/seg_train/seg_train'
test_dir = '/kaggle/input/intel-image-classification/seg_test/seg_test'
batch_size = 32
image_size = (224, 224, 3)
num_classes = 6
epochs = 30

# === LOAD DATA ===
train_gen, val_gen, test_gen, class_names_dict = get_data_generators(train_dir, test_dir, batch_size=batch_size)

# === PREPARE TEST DATA ===
X_test, y_test = next(test_gen.__class__(directory=test_dir,target_size=image_size[:2],batch_size=3000,
                                         class_mode='sparse',shuffle=False))

# === MODEL SETUP ===
model = build_inception_model(num_classes=num_classes, image_size=image_size)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss=["sparse_categorical_crossentropy"] * 3,loss_weights=[1.0, 0.3, 0.3],metrics=["accuracy"] * 3)

# === CALLBACKS ===
callbacks = [keras.callbacks.ModelCheckpoint("best_inception_model.keras", save_best_only=True),
             keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
             keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2)]

# === INITIAL EVALUATION ===
results = model.evaluate([X_test], [y_test, y_test, y_test], verbose=0)
print(f"\nBefore Training Total Loss: {results[0]:.4f}")
print(f"Main Output Loss: {results[1]:.4f} | Accuracy: {results[4]*100:.2f}%")
print(f"Aux1 Output Loss: {results[2]:.4f} | Accuracy: {results[5]*100:.2f}%")
print(f"Aux2 Output Loss: {results[3]:.4f} | Accuracy: {results[6]*100:.2f}%\n")

# === PREPARE TF DATASETS ===
output_signature = (tf.TensorSpec(shape=(None, *image_size), dtype=tf.float32),
                    (tf.TensorSpec(shape=(None,), dtype=tf.float32),
                    tf.TensorSpec(shape=(None,), dtype=tf.float32),
                    tf.TensorSpec(shape=(None,), dtype=tf.float32)))

train_ds = tf.data.Dataset.from_generator(lambda: multi_output_generator(train_gen),output_signature=output_signature)

val_ds = tf.data.Dataset.from_generator(lambda: multi_output_generator(val_gen),output_signature=output_signature)

# === TRAIN MODEL ===
history = model.fit(train_ds,validation_data=val_ds,epochs=epochs,steps_per_epoch=len(train_gen),
                    validation_steps=len(val_gen),callbacks=callbacks)

# === FINAL EVALUATION ===
results = model.evaluate([X_test], [y_test, y_test, y_test], verbose=0)
print(f"\nAfter Training Total Loss: {results[0]:.4f}")
print(f"Main Output Loss: {results[1]:.4f} | Accuracy: {results[4]*100:.2f}%")
print(f"Aux1 Output Loss: {results[2]:.4f} | Accuracy: {results[5]*100:.2f}%")
print(f"Aux2 Output Loss: {results[3]:.4f} | Accuracy: {results[6]*100:.2f}%\n")

# === PLOT TRAINING CURVES ===
pd.DataFrame(history.history).plot(figsize=(10, 6))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.title("Training History")
plt.show()

# === PLOT CONFUSION MATRIX & SAMPLE PREDICTIONS ===
plot_confusion_matrix(model, X_test, y_test, class_names_dict)
plot_predictions(model, X_test, y_test, row=1, col=8, figsize=(15, 3), class_names_dict=class_names_dict)
