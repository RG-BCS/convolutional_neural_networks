# demo_script.py

import matplotlib.pyplot as plt
import pandas as pd
from models import build_mlp_model, build_cnn_model
from dataset_loader import load_fashion_mnist
from utils import plot_predictions

def plot_combined_histories(histories, labels):
    """Plot multiple training histories for comparison."""
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    for history, label in zip(histories, labels):
        plt.plot(history.history['accuracy'], label=f'{label} Train Acc')
        plt.plot(history.history['val_accuracy'], '--', label=f'{label} Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)

    # Plot loss
    plt.subplot(1, 2, 2)
    for history, label in zip(histories, labels):
        plt.plot(history.history['loss'], label=f'{label} Train Loss')
        plt.plot(history.history['val_loss'], '--', label=f'{label} Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def train_and_evaluate(model_builder, model_name, X_train, y_train, X_valid, y_valid, X_test, y_test, input_shape, num_classes, epochs=10, batch_size=32, save_model=False):
    print(f"\n--- Training {model_name} ---")

    model = model_builder(input_shape, num_classes)

    print("Initial test accuracy:")
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"{model_name} test accuracy before training: {acc*100:.2f}%")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    print("Final test accuracy:")
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"{model_name} test accuracy after training: {acc*100:.2f}%")

    if save_model:
        filename = f"{model_name.lower().replace(' ', '_')}_fashion_model.h5"
        model.save(filename)
        print(f"Saved {model_name} to {filename}")

    return model, history

def main():
    epochs = 30
    batch_size = 32
    save_model = True

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test), class_names = load_fashion_mnist()

    input_shape = X_train.shape[1:]
    num_classes = len(class_names)

    # Train MLP
    mlp_model, mlp_history = train_and_evaluate(
        build_mlp_model,
        "MLP Model",
        X_train, y_train, X_valid, y_valid, X_test, y_test,
        input_shape, num_classes,
        epochs=epochs,
        batch_size=batch_size,
        save_model=save_model
    )

    # Train CNN
    cnn_model, cnn_history = train_and_evaluate(
        build_cnn_model,
        "CNN Model",
        X_train, y_train, X_valid, y_valid, X_test, y_test,
        input_shape, num_classes,
        epochs=epochs,
        batch_size=batch_size,
        save_model=save_model
    )

    # Plot combined training histories
    plot_combined_histories([mlp_history, cnn_history], ['MLP', 'CNN'])

    # Plot sample predictions for CNN (best performing model)
    plot_predictions(cnn_model, X_test, y_test, class_names, rows=2, cols=6)

if __name__ == "__main__":
    main()
