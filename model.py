import torch
from torch import nn
from torchvision.models import resnet18

GROUPNORM_GROUPS = 8
CHANNELS = (32, 64, 128)


def _make_norm(norm, num_channels):
    if norm == "batchnorm":
        return nn.BatchNorm2d(num_channels)
    if norm == "groupnorm":
        return nn.GroupNorm(num_groups=GROUPNORM_GROUPS, num_channels=num_channels)
    raise ValueError(norm)


class SmallCNN(nn.Module):
    def __init__(self, num_classes, in_channels=3, normalization="batchnorm", channels=CHANNELS):
        super().__init__()
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=True),
            _make_norm(normalization, c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=True),
            _make_norm(normalization, c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=True),
            _make_norm(normalization, c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c3, num_classes)

    def forward(self, x):
        features = self.features(x)
        pooled = self.global_pool(features)
        flattened = torch.flatten(pooled, 1)
        return self.classifier(flattened)


def build_resnet18_small_input(num_classes, in_channels=3):
    m = resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def build_model(cell, n_classes):
    if cell["model"] == "small_cnn":
        return SmallCNN(num_classes=n_classes, normalization=cell["normalization"])
    if cell["model"] == "resnet18":
        return build_resnet18_small_input(num_classes=n_classes)
    raise ValueError(cell["model"])


def has_batchnorm(model):
    return any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())
