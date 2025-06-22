import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

def get_data_transforms():
    """
    Returns:
        train_transforms (torchvision.transforms.Compose): Transformations for training data.
        valid_transforms (torchvision.transforms.Compose): Transformations for validation data.
    """
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4821, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    valid_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4821, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    return train_transforms, valid_transforms

def get_dataloaders(batch_size=50, num_workers=2, download=True, data_root="./"):
    """
    Prepares CIFAR-10 training and validation dataloaders.

    Args:
        batch_size (int): Batch size for both loaders.
        num_workers (int): Number of subprocesses for data loading.
        download (bool): Whether to download the dataset.
        data_root (str): Path to download or locate the CIFAR-10 dataset.

    Returns:
        train_dl (DataLoader): Training dataloader.
        valid_dl (DataLoader): Validation dataloader.
        class_names_dict (dict): Mapping of class indices to human-readable class names.
    """
    train_tfms, valid_tfms = get_data_transforms()

    train_ds = torchvision.datasets.CIFAR10(
        data_root, train=True, transform=train_tfms, download=download
    )
    valid_ds = torchvision.datasets.CIFAR10(
        data_root, train=False, transform=valid_tfms, download=download
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names_dict = {
        0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
        5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
    }

    return train_dl, valid_dl, class_names_dict
