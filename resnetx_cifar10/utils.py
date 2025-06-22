import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd

# ----------------------------------------
# Unnormalization helper (for plotting)
# ----------------------------------------
def unnormalize(img_tensor):
    """Unnormalize a single image tensor (CHW format) for visualization."""
    mean = torch.tensor([0.4914, 0.4821, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    return img_tensor * std + mean


# ----------------------------------------
# Confusion Matrix Plotting
# ----------------------------------------
def plot_confusion_matrix(model, dataloader, class_names_dict, device):
    """
    Plots a confusion matrix given a model and dataloader.
    """
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y_batch.numpy())

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names_dict.values())
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()


# ----------------------------------------
# Prediction Visualization (Correct + Incorrect)
# ----------------------------------------
def plot_predictions(model, dataloader, class_names_dict, device, row=1, col=8, figsize=(15, 3), max_size=20):
    """
    Visualizes correctly and incorrectly predicted images from the dataset.
    """
    X_correct, X_wrong = [], []
    correct_pred_indx, wrong_pred_indx = [], []
    ground_truth = []

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            preds = model(x_batch.to(device)).argmax(dim=1).cpu()
            for i, (pred, true) in enumerate(zip(preds, y_batch)):
                if len(X_correct) < max_size and pred == true:
                    X_correct.append(x_batch[i].cpu())
                    correct_pred_indx.append(pred.item())
                elif len(X_wrong) < max_size and pred != true:
                    X_wrong.append(x_batch[i].cpu())
                    wrong_pred_indx.append(pred.item())
                    ground_truth.append(true.item())
                if len(X_correct) == max_size and len(X_wrong) == max_size:
                    break
            if len(X_correct) == max_size and len(X_wrong) == max_size:
                break

    # Plot wrong predictions
    print('\n' + '=' * 20 + ' WRONG CLASSIFICATIONS ' + '=' * 20 + '\n')
    _, axs = plt.subplots(row, col, figsize=figsize)
    axs = axs.flatten()
    for i, (pred, true) in enumerate(zip(wrong_pred_indx, ground_truth)):
        img = unnormalize(X_wrong[i]).permute(1, 2, 0).numpy().clip(0, 1)
        axs[i].imshow(img)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlabel(f'Pred: {class_names_dict[pred]}')
        axs[i].set_title(f'True: {class_names_dict[true]}')
    plt.tight_layout()
    plt.show()

    # Plot correct predictions
    print('\n' + '=' * 20 + ' CORRECT CLASSIFICATIONS ' + '=' * 20 + '\n')
    _, axs = plt.subplots(row, col, figsize=figsize)
    axs = axs.flatten()
    for i, pred in enumerate(correct_pred_indx):
        img = unnormalize(X_correct[i]).permute(1, 2, 0).numpy().clip(0, 1)
        axs[i].imshow(img)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlabel(f'Pred: {class_names_dict[pred]}')
        axs[i].set_title(f'True: {class_names_dict[pred]}')
    plt.tight_layout()
    plt.show()


# ----------------------------------------
# Training Loop
# ----------------------------------------
def train_model(model, train_dl, valid_dl, loss_fn, optimizer, num_epochs, device):
    """
    Standard training loop with accuracy and loss tracking.
    """
    train_loss, test_loss = [0.] * num_epochs, [0.] * num_epochs
    train_acc, test_acc = [0.] * num_epochs, [0.] * num_epochs

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss[epoch] += loss.item() * y_batch.size(0)
            correct = (preds.argmax(dim=1).cpu() == y_batch.cpu()).float().sum()
            train_acc[epoch] += correct

        train_loss[epoch] /= len(train_dl.dataset)
        train_acc[epoch] /= len(train_dl.dataset)

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                preds = model(x_batch)
                loss = loss_fn(preds, y_batch)

                test_loss[epoch] += loss.item() * y_batch.size(0)
                correct = (preds.argmax(dim=1).cpu() == y_batch.cpu()).float().sum()
                test_acc[epoch] += correct

        test_loss[epoch] /= len(valid_dl.dataset)
        test_acc[epoch] /= len(valid_dl.dataset)

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1:03d}/{num_epochs:03d} | "
                  f"Train Loss: {train_loss[epoch]:.4f}, Acc: {train_acc[epoch]:.4f} | "
                  f"Val Loss: {test_loss[epoch]:.4f}, Acc: {test_acc[epoch]:.4f}")

    return train_loss, train_acc, test_loss, test_acc
