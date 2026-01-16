import torch
import cv2
from torch.utils.data import Dataset

class HybridAnomalyDataset(Dataset):
    def __init__(self, df, features_array, transform=None):
        self.df = df
        self.features = features_array
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row['processed_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
            
        handcrafted = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(row['anomaly_label'], dtype=torch.float32)
        
        return image, handcrafted, label
