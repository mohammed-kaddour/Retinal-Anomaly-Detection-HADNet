import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

class DeepOnlyAnomalyNet(nn.Module):
    """
    Baseline Model for Ablation Study.
    Uses only deep semantic features from a pre-trained ResNet18, 
    without handcrafted texture features.
    """
    def __init__(self, projection_dim: int = 128):
        super(DeepOnlyAnomalyNet, self).__init__()
        
        # Load pre-trained ResNet18
        resnet = models.resnet18(pretrained=True)
        # Extract features from the average pooling layer (512-dim)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) 
        
        # Projection Head to the 128-D latent space
        self.projection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for deep feature extraction and projection.
        """
        # Feature extraction: output shape [Batch, 512, 1, 1]
        deep_feat = self.backbone(images)
        
        # Project to latent space: output shape [Batch, 128]
        projected = self.projection_head(deep_feat)
        
        return projected
