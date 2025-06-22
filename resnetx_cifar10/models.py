import torch
import torch.nn as nn

# -------------------------------
# Basic pre-activation residual block (ResNet18, ResNet34)
# -------------------------------
class PreactivationBlock(nn.Module):
    expansion = 1  # used to compute output channels

    def __init__(self, in_slices, slices, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_slices)
        self.conv1 = nn.Conv2d(in_slices, slices, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(slices)
        self.conv2 = nn.Conv2d(slices, slices, kernel_size=3, stride=1, padding=1, bias=False)

        # Shortcut for changing spatial resolution or channel depth
        if stride != 1 or in_slices != self.expansion * slices:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_slices, self.expansion * slices, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else out

        out = self.conv1(out)
        out = self.conv2(nn.ReLU()(self.bn2(out)))
        out += shortcut
        return out

# -------------------------------
# Bottleneck pre-activation block (ResNet50+)
# -------------------------------
class PreactivationBottleneckBlock(nn.Module):
    expansion = 4  # Bottleneck increases feature map depth

    def __init__(self, in_slices, slices, stride=1):
        super().__init__()

        self.b1 = nn.BatchNorm2d(in_slices)
        self.conv1 = nn.Conv2d(in_slices, slices, kernel_size=1, bias=False)

        self.b2 = nn.BatchNorm2d(slices)
        self.conv2 = nn.Conv2d(slices, slices, kernel_size=3, stride=stride, padding=1, bias=False)

        self.b3 = nn.BatchNorm2d(slices)
        self.conv3 = nn.Conv2d(slices, self.expansion * slices, kernel_size=1, bias=False)

        if stride != 1 or in_slices != self.expansion * slices:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_slices, self.expansion * slices, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = nn.ReLU()(self.b1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x

        out = self.conv1(out)
        out = nn.ReLU()(self.b2(out))
        out = self.conv2(out)
        out = nn.ReLU()(self.b3(out))
        out = self.conv3(out)

        out += shortcut
        return out

# -------------------------------
# Pre-activation ResNet base class
# -------------------------------
class PreActResNet(nn.Module):
    def __init__(self, block_type, num_blocks, num_classes=10):
        super().__init__()
        self.in_slices = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(3, self.in_slices, kernel_size=3, stride=1, padding=1, bias=False)

        # Residual layers
        self.layer1 = self._make_layer(block_type, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block_type, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block_type, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block_type, 512, num_blocks[3], stride=2)

        # Classification head
        self.linear = nn.Linear(512 * block_type.expansion, num_classes)

    def _make_layer(self, block, slices, num_blocks, stride):
        """Creates a residual stage with multiple blocks"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_slices, slices, s))
            self.in_slices = block.expansion * slices
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        out = nn.Flatten()(out)
        return self.linear(out)

# -------------------------------
# Named constructors for ResNet variants
# -------------------------------
def PreActResNet18():
    return PreActResNet(PreactivationBlock, [2, 2, 2, 2])

def PreActResNet34():
    return PreActResNet(PreactivationBlock, [3, 4, 6, 3])

def PreActResNet50():
    return PreActResNet(PreactivationBottleneckBlock, [3, 4, 6, 3])

def PreActResNet101():
    return PreActResNet(PreactivationBottleneckBlock, [3, 4, 23, 3])

def PreActResNet152():
    return PreActResNet(PreactivationBottleneckBlock, [3, 8, 36, 3])
