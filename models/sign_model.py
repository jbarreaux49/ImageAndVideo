"""R3D-18 backbone with a custom classification head for ASL recognition."""
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class SignModel(nn.Module):
    """Pretrained R3D-18 with dropout + linear head for `num_classes` ASL signs."""

    def __init__(self, num_classes: int, dropout_rate: float = 0.5):
        super().__init__()
        backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)

        # Reuse all layers except the original FC classifier
        self.features   = nn.Sequential(*list(backbone.children())[:-1])
        self.dropout    = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        return self.classifier(x)
