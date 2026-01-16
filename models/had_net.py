import torch
import torch.nn as nn
import torchvision.models as models

# --- PROPOSED HYBRID MODEL ---
class HybridAnomalyNet(nn.Module):
    def __init__(self, handcrafted_input_size=16, projection_dim=128):
        super(HybridAnomalyNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) 
        
        self.texture_net = nn.Sequential(
            nn.Linear(handcrafted_input_size, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        self.fusion_layer = nn.Linear(512 + 32, 256)
        self.projection_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )
        
    def forward(self, images, handcrafted):
        deep_feat = self.backbone(images).view(images.size(0), -1)
        text_feat = self.texture_net(handcrafted)
        combined = torch.cat((deep_feat, text_feat), dim=1)
        return self.projection_head(self.fusion_layer(combined))

# --- BASELINE MODEL ---
class DeepOnlyAnomalyNet(nn.Module):
    def __init__(self, projection_dim=128):
        super(DeepOnlyAnomalyNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) 
        self.projection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )
        
    def forward(self, images):
        deep_feat = self.backbone(images)
        return self.projection_head(deep_feat)
