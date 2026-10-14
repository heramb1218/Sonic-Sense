"""
CNN Audio Classifier Architecture

A 4-block 2D Convolutional Neural Network designed for classifying
Log-Mel Spectrogram representations of urban acoustic events.

Input shape:
    (batch_size, 1, n_mels=128, time_frames=173)

Output:
    Raw logits for each sound category (batch_size, num_classes=10).
    Softmax is applied during evaluation and inference for probabilities.

Usage:
    import torch
    from src.models.cnn import UrbanSoundCNN

    model = UrbanSoundCNN(num_classes=10)
    dummy_input = torch.randn(4, 1, 128, 173)
    logits = model(dummy_input)  # Shape: (4, 10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block: Conv2D -> BatchNorm2d -> ReLU -> MaxPool2d -> Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool_size: tuple[int, int] = (2, 2),
        dropout: float = 0.25,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class UrbanSoundCNN(nn.Module):
    """Convolutional Neural Network for Urban Sound Classification."""

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.num_classes = num_classes

        # Feature extractor
        # Input: (B, 1, 128, 173)
        self.block1 = ConvBlock(in_channels, 32, kernel_size=3, padding=1, pool_size=(2, 2), dropout=0.2)
        # Shape after block1: (B, 32, 64, 86)

        self.block2 = ConvBlock(32, 64, kernel_size=3, padding=1, pool_size=(2, 2), dropout=0.2)
        # Shape after block2: (B, 64, 32, 43)

        self.block3 = ConvBlock(64, 128, kernel_size=3, padding=1, pool_size=(2, 2), dropout=0.25)
        # Shape after block3: (B, 128, 16, 21)

        self.block4 = ConvBlock(128, 256, kernel_size=3, padding=1, pool_size=(2, 2), dropout=0.25)
        # Shape after block4: (B, 256, 8, 10)

        # Global average pooling produces fixed (B, 256, 1, 1) regardless of exact input length
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head
        self.fc1 = nn.Linear(256, 128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 1, n_mels, time_frames).

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute predicted class probabilities via Softmax."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
