# utils.py
"""
Utility functions for model evaluation and data generation.
Includes functions for plotting predictions, confusion matrix,
and creating multi-output data generators.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf

# Default class names dict - can be overridden by passing a custom one to the functions
DEFAULT_CLASS_NAMES_DICT = {
    0: 'buildings',
    1: 'forest',
    2: 'glacier',
    3: 'mountain',
    4: 'sea',
    5: 'street'
}

def plot_predictions(model, X_test, y_test, row=4, col=10, figsize=(6, 4), class_names_dict=DEFAULT_CLASS_NAMES_DICT):
    """
    Plot a grid of correct and wrong predictions made by the model on test data.

    Args:
        model: Trained Keras model with multiple outputs.
        X_test: Test images.
        y_test: True labels for test images.
        row: Number of rows in plot.
        col: Number of columns in plot.
        figsize: Figure size for matplotlib plots.
        class_names_dict: Dict mapping class indices to class names. Defaults to DEFAULT_CLASS_NAMES_DICT.
    """
    indices = np.arange(len(X_test))
    np.random.shuffle(indices)
    X_test = X_test[indices]
    y_test = y_test[indices]

    prob = model.predict([X_test], verbose=0)[0]  # Main output predictions
    pred_labels = prob.argmax(axis=1)
    correct_pred_indx = np.where(pred_labels == y_test)[0]
    wrong_pred_indx = np.where(pred_labels != y_test)[0]

    print('\t\t\t\t\t\t WRONG CLASSIFICATIONS\n')
    _, axs = plt.subplots(row, col, figsize=figsize)
    axs = axs.flatten()
    for i, wrong_indx in zip(range(len(axs)), wrong_pred_indx):
        axs[i].imshow(X_test[wrong_indx])
        axs[i].set_xticks(())
        axs[i].set_yticks(())
        axs[i].set_xlabel(f'Pred| {class_names_dict.get(pred_labels[wrong_indx], "Unknown")}')
        axs[i].set_title(f'GT| {class_names_dict.get(y_test[wrong_indx], "Unknown")}')
    plt.show()

    print('\t\t\t\t\t\t CORRECT CLASSIFICATIONS\n')
    _, axs = plt.subplots(row, col, figsize=figsize)
    axs = axs.flatten()
    for i, correct_indx in zip(range(len(axs)), correct_pred_indx):
        axs[i].imshow(X_test[correct_indx])
        axs[i].set_xticks(())
        axs[i].set_yticks(())
        axs[i].set_xlabel(f'Pred| {class_names_dict.get(pred_labels[correct_indx], "Unknown")}')
        axs[i].set_title(f'GT| {class_names_dict.get(y_test[correct_indx], "Unknown")}')
    plt.show()

def plot_confusion_matrix(model, X_test, y_test, class_names_dict=DEFAULT_CLASS_NAMES_DICT):
    """
    Plot the confusion matrix for model predictions on test data.

    Args:
        model: Trained Keras model with multiple outputs.
        X_test: Test images.
        y_test: True labels for test images.
        class_names_dict: Dict mapping class indices to class names. Defaults to DEFAULT_CLASS_NAMES_DICT.
    """
    prob = model.predict([X_test], verbose=0)[0]  # Main output predictions
    pred_labels = prob.argmax(axis=1)

    cm = confusion_matrix(y_test, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[class_names_dict.get(i, "Unknown") for i in range(len(class_names_dict))])
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()

def multi_output_generator(gen):
    """
    Generator wrapper to yield data compatible with multi-output model training.

    Args:
        gen: A keras data generator yielding (inputs, labels).

    Yields:
        inputs, tuple of (labels, labels, labels) for main and auxiliary outputs.
    """
    for x, y in gen:
        yield x, (y, y, y)  # 3 outputs: main, aux1, aux2
