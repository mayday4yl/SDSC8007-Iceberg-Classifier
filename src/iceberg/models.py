from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.hub import load_state_dict_from_url


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(channels, channels),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class IcebergCNN(nn.Module):
    """Small multimodal CNN for 75x75 SAR images plus tabular auxiliary features."""

    def __init__(
        self,
        in_channels: int = 4,
        angle_dim: int = 2,
        width: int = 32,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = width, width * 2, width * 4, width * 6
        self.image_net = nn.Sequential(
            ConvBlock(in_channels, c1),
            ResidualBlock(c1),
            nn.AvgPool2d(2),
            ConvBlock(c1, c2),
            ResidualBlock(c2),
            nn.AvgPool2d(2),
            ConvBlock(c2, c3),
            ResidualBlock(c3),
            nn.AvgPool2d(2),
            ConvBlock(c3, c4),
            ResidualBlock(c4),
            nn.AdaptiveAvgPool2d(1),
        )
        aux_width = max(16, min(64, angle_dim * 2))
        self.angle_net = nn.Sequential(
            nn.Linear(angle_dim, aux_width),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(aux_width, 16),
            nn.SiLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(c4 + 16, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, images: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        image_features = self.image_net(images).flatten(1)
        angle_features = self.angle_net(angles)
        features = torch.cat([image_features, angle_features], dim=1)
        return self.classifier(features).squeeze(1)


class IcebergVGG(nn.Module):
    """VGG-style CNN inspired by common public Statoil solution patterns."""

    def __init__(
        self,
        in_channels: int = 4,
        angle_dim: int = 2,
        width: int = 32,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = width, width * 2, width * 4, width * 6

        def stage(in_ch: int, out_ch: int, drop: float) -> nn.Sequential:
            return nn.Sequential(
                ConvBlock(in_ch, out_ch),
                ConvBlock(out_ch, out_ch),
                nn.MaxPool2d(2),
                nn.Dropout2d(drop),
            )

        self.image_net = nn.Sequential(
            stage(in_channels, c1, dropout * 0.20),
            stage(c1, c2, dropout * 0.25),
            stage(c2, c3, dropout * 0.30),
            ConvBlock(c3, c4),
            ConvBlock(c4, c4),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        aux_width = max(16, min(64, angle_dim * 2))
        self.angle_net = nn.Sequential(
            nn.Linear(angle_dim, aux_width),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(aux_width, 16),
            nn.SiLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(c4 * 2 + 16, 256),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, images: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        image_maps = self.image_net(images)
        image_features = torch.cat([self.avg_pool(image_maps).flatten(1), self.max_pool(image_maps).flatten(1)], dim=1)
        angle_features = self.angle_net(angles)
        features = torch.cat([image_features, angle_features], dim=1)
        return self.classifier(features).squeeze(1)


class ResNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


def _make_resnet_layer(inplanes: int, planes: int, blocks: int, stride: int) -> nn.Sequential:
    layers = [ResNetBasicBlock(inplanes, planes, stride=stride)]
    for _ in range(1, blocks):
        layers.append(ResNetBasicBlock(planes, planes, stride=1))
    return nn.Sequential(*layers)


class PretrainedFiLMResNet(nn.Module):
    """ImageNet ResNet34 backbone with incidence-angle FiLM modulation.

    FiLM is initialized as an identity transform (`gamma = 1`, `beta = 0`) so
    the pretrained backbone is not randomly distorted at the start of training.
    """

    def __init__(
        self,
        in_channels: int = 4,
        angle_dim: int = 2,
        width: int = 32,
        dropout: float = 0.35,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = _make_resnet_layer(64, 64, blocks=3, stride=1)
        self.layer2 = _make_resnet_layer(64, 128, blocks=4, stride=2)
        self.layer3 = _make_resnet_layer(128, 256, blocks=6, stride=2)
        self.layer4 = _make_resnet_layer(256, 512, blocks=3, stride=2)

        self.film_gen = nn.Sequential(
            nn.Linear(angle_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1024),
        )
        nn.init.zeros_(self.film_gen[-1].weight)
        nn.init.zeros_(self.film_gen[-1].bias)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )
        if pretrained:
            self.load_resnet34_weights(in_channels)

    def load_resnet34_weights(self, in_channels: int) -> None:
        state = load_state_dict_from_url(
            "https://download.pytorch.org/models/resnet34-b627a593.pth",
            progress=True,
        )
        conv1_weight = state.pop("conv1.weight")
        state.pop("fc.weight", None)
        state.pop("fc.bias", None)
        missing, unexpected = self.load_state_dict(state, strict=False)
        allowed_missing = {
            "conv1.weight",
            "film_gen.0.weight",
            "film_gen.0.bias",
            "film_gen.2.weight",
            "film_gen.2.bias",
            "classifier.1.weight",
            "classifier.1.bias",
            "classifier.4.weight",
            "classifier.4.bias",
        }
        unknown_missing = set(missing) - allowed_missing
        if unknown_missing or unexpected:
            raise RuntimeError(f"Unexpected ResNet34 weight loading result: missing={missing}, unexpected={unexpected}")
        self._init_first_conv(conv1_weight, in_channels)

    def _init_first_conv(self, weight: torch.Tensor, in_channels: int) -> None:
        with torch.no_grad():
            if in_channels == 3:
                self.conv1.weight.copy_(weight)
            elif in_channels > 3:
                self.conv1.weight[:, :3].copy_(weight)
                mean_channel = weight.mean(dim=1, keepdim=True)
                self.conv1.weight[:, 3:].copy_(mean_channel.repeat(1, in_channels - 3, 1, 1))
            else:
                self.conv1.weight.copy_(weight[:, :in_channels] * (3.0 / float(in_channels)))

    def forward(self, images: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        film_params = self.film_gen(angles)
        gamma_delta, beta = film_params.chunk(2, dim=1)
        gamma = 1.0 + gamma_delta.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta

        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x).squeeze(1)


def build_model(
    arch: str,
    in_channels: int,
    angle_dim: int,
    width: int,
    dropout: float,
    pretrained: bool = True,
) -> nn.Module:
    if arch == "resnet":
        return IcebergCNN(in_channels=in_channels, angle_dim=angle_dim, width=width, dropout=dropout)
    if arch == "vgg":
        return IcebergVGG(in_channels=in_channels, angle_dim=angle_dim, width=width, dropout=dropout)
    if arch == "film_resnet":
        return PretrainedFiLMResNet(
            in_channels=in_channels,
            angle_dim=angle_dim,
            width=width,
            dropout=dropout,
            pretrained=pretrained,
        )
    raise ValueError(f"Unknown architecture: {arch}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
