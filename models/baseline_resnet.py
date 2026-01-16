import torch
import torch.nn as nn
import torchvision.models as models

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
