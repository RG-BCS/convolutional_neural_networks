import torch
import copy
import torch.nn as nn
import torchvision.models as models
from resnet_models import initialize_resnet34
from dataloader_generator import get_dataloaders
from utils import (
    train_model,
    plot_loss_accuracy,
    plot_confusion_matrix,
    plot_predictions
)

def transfer_learning_feature_extraction(model, train_dl, valid_dl, loss_fn, optimizer, num_epochs, device):
    """
    Freezes base layers of the model; trains only the final classification head.
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    print("\n\t\tTransfer Learning: Feature Extraction\n")
    train_loss, train_acc, valid_loss, valid_acc = train_model(
        model, train_dl, valid_dl, loss_fn, optimizer, num_epochs, device
    )

    plot_loss_accuracy(
        {"train_loss": train_loss, "valid_loss": valid_loss},
        {"train_accu": train_acc, "valid_accu": valid_acc}
    )
    return model

def transfer_learning_fine_tuning(model, train_dl, valid_dl, loss_fn, optimizer, num_epochs, device):
    """
    Trains all model parameters including pretrained layers.
    """
    print("\n\n\t\tTransfer Learning: Fine-Tuning\n")
    train_loss, train_acc, valid_loss, valid_acc = train_model(
        model, train_dl, valid_dl, loss_fn, optimizer, num_epochs, device
    )

    plot_loss_accuracy(
        {"train_loss": train_loss, "valid_loss": valid_loss},
        {"train_accu": train_acc, "valid_accu": valid_acc}
    )
    return model

if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(42)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 50
    num_epochs = 25
    lr = 1e-4

    # Load data
    train_dl, valid_dl, class_names_dict = get_dataloaders(batch_size=batch_size)

    # Load and prepare model
    model_fe = initialize_resnet34(num_classes=len(class_names_dict), pretrained=True)
    model_fe.to(device)
    model_ft = copy.deepcopy(model_fe)

    loss_fn = nn.CrossEntropyLoss()
    optimizer_fe = torch.optim.Adam(model_fe.fc.parameters(), lr=lr)
    optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=lr)

    # Run transfer learning: feature extraction
    model_fe = transfer_learning_feature_extraction(
        model_fe, train_dl, valid_dl, loss_fn, optimizer_fe, num_epochs, device
    )
    plot_confusion_matrix(model_fe, valid_dl, class_names_dict, device)
    plot_predictions(model_fe, train_dl, class_names_dict, device)

    # Run transfer learning: fine-tuning
    model_ft = transfer_learning_fine_tuning(
        model_ft, train_dl, valid_dl, loss_fn, optimizer_ft, num_epochs, device
    )
    plot_confusion_matrix(model_ft, valid_dl, class_names_dict, device)
    plot_predictions(model_ft, train_dl, class_names_dict, device)
