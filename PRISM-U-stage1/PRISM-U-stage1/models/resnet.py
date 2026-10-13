import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18_cifar(num_classes=10):
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
