import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --------------------------------------------
# Plot a confusion matrix of model predictions
# --------------------------------------------
def plot_confusion_matrix(model, dataloader, class_names_dict, device):
    """
    Plots the confusion matrix for the model's predictions on the validation dataset.
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
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_names_dict.values()))
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()


# ------------------------------------------------------------------
# Visualize correct and incorrect predictions on a batch of samples
# ------------------------------------------------------------------
def plot_predictions(model, dataloader, class_names_dict, device, row=1, col=8, figsize=(15, 3), max_size=20):
    """
    Plots example predictions: first the incorrect ones, then correct ones.
    Shows predicted and ground truth labels for visual inspection.
    """
    def unnormalize(img_tensor):
        """Unnormalizes an image tensor using CIFAR-10 dataset mean and std."""
        mean = torch.tensor([0.4914, 0.4821, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
        return img_tensor * std + mean

    X_correct, X_wrong = [], []
    correct_pred_indx, wrong_pred_indx, ground_truth = [], [], []

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            preds = model(x_batch.to(device)).argmax(dim=1).cpu()
            for i, (pred, label) in enumerate(zip(preds, y_batch)):
                if len(X_correct) < max_size and pred == label:
                    X_correct.append(x_batch[i].cpu())
                    correct_pred_indx.append(pred.item())
                elif len(X_wrong) < max_size and pred != label:
                    X_wrong.append(x_batch[i].cpu())
                    wrong_pred_indx.append(pred.item())
                    ground_truth.append(label.item())
                if len(wrong_pred_indx) == max_size and len(correct_pred_indx) == max_size:
                    break
            if len(wrong_pred_indx) == max_size and len(correct_pred_indx) == max_size:
                break

    # Plot wrong predictions
    print('\t\t\t\tWRONG CLASSIFICATIONS\n')
    _, axs = plt.subplots(row, col, figsize=figsize)
    axs = axs.flatten()
    for i, pred_idx, true_idx in zip(range(len(axs)), wrong_pred_indx, ground_truth):
        img = unnormalize(X_wrong[i]).permute(1, 2, 0).numpy().clip(0, 1)
        axs[i].imshow(img)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlabel(f'Pred: {class_names_dict[pred_idx]}')
        axs[i].set_title(f'True: {class_names_dict[true_idx]}')
    plt.tight_layout()
    plt.show()

    # Plot correct predictions
    print('\t\t\t\tCORRECT CLASSIFICATIONS\n')
    _, axs = plt.subplots(row, col, figsize=figsize)
    axs = axs.flatten()
    for i, pred_idx in zip(range(len(axs)), correct_pred_indx):
        img = unnormalize(X_correct[i]).permute(1, 2, 0).numpy().clip(0, 1)
        axs[i].imshow(img)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlabel(f'Pred: {class_names_dict[pred_idx]}')
        axs[i].set_title(f'True: {class_names_dict[pred_idx]}')
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------
# Core training function (used in both TL modes)
# --------------------------------------------------------
def train_model(model, train_dl, valid_dl, loss_fn, optimizer, num_epochs, device):
    """
    Trains the model and returns loss and accuracy per epoch for both training and validation.
    """
    train_loss, test_loss = [0.] * num_epochs, [0.] * num_epochs
    train_acc, test_acc = [0.] * num_epochs, [0.] * num_epochs

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss[epoch] += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1).cpu() == y_batch.cpu()).float().sum()
            train_acc[epoch] += is_correct

        train_loss[epoch] /= len(train_dl.dataset)
        train_acc[epoch] /= len(train_dl.dataset)

        # Validation
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)

                test_loss[epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1).cpu() == y_batch.cpu()).float().sum()
                test_acc[epoch] += is_correct

        test_loss[epoch] /= len(valid_dl.dataset)
        test_acc[epoch] /= len(valid_dl.dataset)

        # Print progress every 5 epochs
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1:03d}/{num_epochs:03d}| "
                  f"train_loss: {train_loss[epoch]:.4f}| train_acc: {train_acc[epoch]:.4f}| "
                  f"valid_loss: {test_loss[epoch]:.4f}| valid_acc: {test_acc[epoch]:.4f}")

    return train_loss, train_acc, test_loss, test_acc


# -----------------------------------------------------------------
# Visualize training loss and accuracy over epochs
# -----------------------------------------------------------------
def plot_loss_accuracy(losses, accuracies):
    """
    Plots training and validation loss and accuracy curves.
    Expects dictionary inputs with lists of values per epoch.
    """
    loss_df = pd.DataFrame({k: list(map(float, v)) for k, v in losses.items()})
    acc_df = pd.DataFrame({k: list(map(float, v)) for k, v in accuracies.items()})

    plt.figure(figsize=(10, 4))
    loss_df.plot(ax=plt.subplot(1, 2, 1), title="Loss Curves")
    plt.grid(True)
    acc_df.plot(ax=plt.subplot(1, 2, 2), title="Accuracy Curves")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
