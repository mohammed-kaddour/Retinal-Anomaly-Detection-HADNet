# Imports corrigés
import torch
from torch.utils.data import DataLoader
from models.had_net import HybridAnomalyNet
from models.baseline_resnet import DeepOnlyAnomalyNet
from data.dataset import HybridAnomalyDataset
from utils.trainer import init_center, train_one_epoch

# Exemple d'utilisation pour le modèle Hybride
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridAnomalyNet().to(device)
# ... initialisation dataset et loader ...
center = init_center(model, train_loader, device, is_hybrid=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(20):
    loss = train_one_epoch(model, train_loader, center, optimizer, device, is_hybrid=True)
    print(f"Epoch {epoch} Loss: {loss:.4f}")
