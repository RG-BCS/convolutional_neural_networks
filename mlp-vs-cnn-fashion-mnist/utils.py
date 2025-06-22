# utils/plotting.py

import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(model, X, y_true, class_names, rows=2, cols=6, figsize=(12, 5), show_correct=True, show_incorrect=True):
    """
    Plots correct and/or incorrect predictions made by the model.

    Parameters:
    - model: Trained Keras model.
    - X: Input images (must match model input shape).
    - y_true: Ground truth labels.
    - class_names: List of class label names.
    - rows, cols: Number of rows and columns in the subplot grid.
    - figsize: Size of each figure.
    - show_correct: Whether to display correctly classified images.
    - show_incorrect: Whether to display incorrectly classified images.
    """
    probs = model.predict(X, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    correct_indices = np.where(y_pred == y_true)[0]
    incorrect_indices = np.where(y_pred != y_true)[0]

    def _plot(indices, title):
        print(f"\n{title.upper()}")
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        axs = axs.flatten()

        for i, idx in enumerate(indices[:rows * cols]):
            axs[i].imshow(X[idx].squeeze(), cmap="gray")
            axs[i].axis("off")
            axs[i].set_title(f"Pred: {class_names[y_pred[idx]]}\nGT: {class_names[y_true[idx]]}", fontsize=8)

        plt.tight_layout()
        plt.show()

    if show_incorrect and len(incorrect_indices) > 0:
        _plot(incorrect_indices, "Incorrect Predictions")

    if show_correct and len(correct_indices) > 0:
        _plot(correct_indices, "Correct Predictions")
