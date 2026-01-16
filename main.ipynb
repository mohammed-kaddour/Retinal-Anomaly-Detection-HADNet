# 1. Imports
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from data.dataset import HybridAnomalyDataset
from models.had_net import HybridAnomalyNet
from utils.trainer import init_center, train_model

# 2. Config & Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Transforms (ImageNet standard)
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Data Loading (Assume anomaly_train_df and train_feats are ready)
train_dataset = HybridAnomalyDataset(anomaly_train_df, train_feats, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 5. Model Setup
model = HybridAnomalyNet(handcrafted_input_size=16).to(device)

# 6. Initialize Center & Train
center = init_center(model, train_loader, device)
losses = train_model(model, train_loader, center, device, epochs=20)

# 7. Save
torch.save({'model_state': model.state_dict(), 'center': center}, "had_net_final.pth")
