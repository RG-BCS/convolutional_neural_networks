import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

def get_cifar10_dataloaders(batch_size=100, num_workers=2, augment=True):
    """
    Returns training and validation DataLoaders for CIFAR-10.
    
    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        augment (bool): Whether to apply data augmentation to training data.
    
    Returns:
        train_dl (DataLoader): DataLoader for training data.
        valid_dl (DataLoader): DataLoader for validation/test data.
        class_names_dict (dict): Mapping from class index to class name.
    """
    # Normalization parameters from CIFAR-10 statistics
    mean = (0.4914, 0.4821, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # Data augmentation and normalization for training
    if augment:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    # Only normalization for validation
    valid_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Datasets
    train_ds = torchvision.datasets.CIFAR10(root='./', train=True, transform=train_transforms, download=True)
    valid_ds = torchvision.datasets.CIFAR10(root='./', train=False, transform=valid_transforms, download=True)

    # DataLoaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Class index to label name
    class_names_dict = {
        0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
        5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
    }

    return train_dl, valid_dl, class_names_dict
