import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet34_Weights

def get_resnet34_feature_extractor(num_classes: int):
    """
    Load pretrained ResNet34 and freeze all layers except the final fully connected layer.
    This is for feature extraction (only the last layer is trained).
    """
    model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace final layer for CIFAR-10 (or any num_classes)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    # Only the final layer will have requires_grad = True
    return model

def get_resnet34_fine_tuning(num_classes: int):
    """
    Load pretrained ResNet34 and make the entire model trainable.
    This is for fine-tuning (the whole network is updated).
    """
    model = models.resnet34(weights=ResNet34_Weights.DEFAULT)

    # Replace final layer for CIFAR-10 (or any num_classes)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    # All layers will be trainable
    return model
