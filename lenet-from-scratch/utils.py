import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_training_curves(history):
    """Plots training and validation accuracy/loss curves.

    Args:
        history (History): Keras History object returned by model.fit()
    """
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.title("Training & Validation Accuracy/Loss")
    plt.xlabel("Epochs")
    plt.show()


def plot_predictions(model, X_test, y_test, row=2, col=5, figsize=(10, 5)):
    """Displays a grid of correct and incorrect model predictions.

    Args:
        model (keras.Model): Trained model.
        X_test (np.ndarray): Test images.
        y_test (np.ndarray): Ground truth labels.
        row (int): Number of rows for each grid (correct/incorrect).
        col (int): Number of columns for each grid.
        figsize (tuple): Figure size for each plot.
    """
    prob = model.predict(X_test, verbose=0)
    pred_labels = prob.argmax(axis=1)

    correct_idx = np.where(pred_labels == y_test)[0]
    wrong_idx = np.where(pred_labels != y_test)[0]

    # Wrong predictions
    print("\nWRONG CLASSIFICATIONS\n")
    _, axs = plt.subplots(row, col, figsize=figsize)
    axs = axs.flatten()
    for i, idx in zip(range(len(axs)), wrong_idx):
        axs[i].imshow(X_test[idx].squeeze(), cmap='Greys')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlabel(f'Pred: {pred_labels[idx]}')
        axs[i].set_title(f'True: {y_test[idx]}')
    plt.tight_layout()
    plt.show()

    # Correct predictions
    print("\nCORRECT CLASSIFICATIONS\n")
    _, axs = plt.subplots(row, col, figsize=figsize)
    axs = axs.flatten()
    for i, idx in zip(range(len(axs)), correct_idx):
        axs[i].imshow(X_test[idx].squeeze(), cmap='Greys')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlabel(f'Pred: {pred_labels[idx]}')
        axs[i].set_title(f'True: {y_test[idx]}')
    plt.tight_layout()
    plt.show()
