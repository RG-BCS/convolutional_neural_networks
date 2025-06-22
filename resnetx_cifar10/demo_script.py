import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

from models import PreActResNet34
from utils import train_model, plot_confusion_matrix, plot_predictions
from dataloader_generator import get_cifar10_dataloaders

def main():
    # Set training parameters
    num_epochs = 50
    batch_size = 100
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_dl, valid_dl, class_names_dict = get_cifar10_dataloaders(batch_size=batch_size)

    # Initialize model
    model = PreActResNet34().to(device)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_loss, train_acc, test_loss, test_acc = train_model(
        model, train_dl, valid_dl, loss_fn, optimizer, num_epochs, device
    )

    # Plot training history
    history = {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'test_loss': test_loss,
        'test_acc': test_acc
    }
    pd.DataFrame(history).plot(figsize=(10, 5), title="Training History")
    plt.grid()
    plt.show()

    # Evaluation: Confusion matrix and prediction samples
    plot_confusion_matrix(model, valid_dl, class_names_dict, device)
    plot_predictions(model, train_dl, class_names_dict, device, row=1, col=8, figsize=(15, 3), max_size=20)

if __name__ == '__main__':
    main()
