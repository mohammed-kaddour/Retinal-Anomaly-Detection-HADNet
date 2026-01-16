import torch
import torch.nn as nn
import torchvision.models as models

class DeepOnlyAnomalyNet(nn.Module):
    """
    Baseline Model: Uses only deep semantic features from ResNet18.
    No texture features (Haralick/LBP) are used here.
    """
    def __init__(self, projection_dim=128):
        super(DeepOnlyAnomalyNet, self).__init__()
        
        # Load standard ResNet18 backbone
        resnet = models.resnet18(pretrained=True)
        # Remove the last classification layer (fc)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) 
        
        # Simple Projection Head (512 -> 256 -> 128)
        # This matches the latent space dimension of the Hybrid model for fair comparison
        self.projection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )
        
    def forward(self, images):
        # 1. Extract deep features
        deep_feat = self.backbone(images)
        
        # 2. Project to latent space
        projected = self.projection_head(deep_feat)
        
        return projected
