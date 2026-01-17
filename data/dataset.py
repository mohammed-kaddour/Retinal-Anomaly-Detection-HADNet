import torch
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple

class HybridAnomalyDataset(Dataset):
    """
    Custom Dataset for Multi-modal Retinal Anomaly Detection.
    Loads processed images and their corresponding handcrafted texture features.
    """
    def __init__(self, df: pd.DataFrame, features_array: np.ndarray, transform: Optional[Callable] = None):
        """
        Args:
            df: Dataframe containing image paths and anomaly labels.
            features_array: Numpy array containing extracted Haralick/LBP features.
            transform: PyTorch transforms for image augmentation/normalization.
        """
        self.df = df
        self.features = features_array
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # Load the preprocessed retinal image
        image = cv2.imread(row['processed_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply torchvision transforms (Resize, Normalize, etc.)
        if self.transform:
            image = self.transform(image)
            
        # Convert handcrafted features (16-dim) to float tensor
        handcrafted = torch.tensor(self.features[idx], dtype=torch.float32)
        
        # Anomaly label (0: Healthy, 1: Anomaly)
        label = torch.tensor(row['anomaly_label'], dtype=torch.float32)
        
        return image, handcrafted, label
