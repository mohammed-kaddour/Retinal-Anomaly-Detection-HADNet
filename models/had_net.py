import torch
import torch.nn as nn
import torchvision.models as models

class HybridAnomalyNet(nn.Module):
    """
    HAD-Net (Hybrid Anomaly Detection Network).
    A Dual-Stream architecture fusing deep semantic features and 
    handcrafted texture biomarkers into a shared latent space.
    """
    def __init__(self, handcrafted_input_size: int = 16, projection_dim: int = 128):
        super(HybridAnomalyNet, self).__init__()
        
        # --- Stream 1: Deep Semantic Features (Pretrained ResNet18) ---
        resnet = models.resnet18(pretrained=True)
        # Remove the final fully connected layer to get 512-dim features
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) 
        
        # --- Stream 2: Handcrafted Texture Features (MLP) ---
        # Processes the 16 texture descriptors into a 32-dim embedding
        self.texture_net = nn.Sequential(
            nn.Linear(handcrafted_input_size, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # --- Fusion and Latent Projection ---
        # Combined size: 512 (Deep) + 32 (Texture) = 544
        self.fusion_layer = nn.Linear(512 + 32, 256)
        
        # Projection head to the hypersphere for One-Class Center Loss (SVDD)
        self.projection_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )
        
    def forward(self, images: torch.Tensor, handcrafted: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for hybrid feature fusion.
        """
        # Extract deep features: [Batch, 512, 1, 1] -> [Batch, 512]
        deep_feat = self.backbone(images).view(images.size(0), -1)
        
        # Extract texture features: [Batch, 32]
        text_feat = self.texture_net(handcrafted)
        
        # Feature Fusion (Concatenation)
        combined = torch.cat((deep_feat, text_feat), dim=1)
        
        # Projection to the 128-D latent space
        fused = self.fusion_layer(combined)
        projected = self.projection_head(fused)
        
        return projected
