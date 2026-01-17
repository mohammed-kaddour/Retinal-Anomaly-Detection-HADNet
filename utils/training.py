import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Union

def init_center(model: torch.nn.Module, loader: DataLoader, device: torch.device, is_hybrid: bool = True) -> torch.Tensor:
    """
    Initializes the center 'c' of the hypersphere for Deep SVDD.
    The center is the mean of the network outputs for the initial pass of healthy data.
    
    Args:
        model: The HAD-Net or Baseline model.
        loader: DataLoader containing only healthy training samples.
        device: CPU or GPU.
        is_hybrid: If True, uses both images and texture features.
        
    Returns:
        The calculated center point in the latent space.
    """
    model.eval()
    center = torch.zeros(128).to(device)
    n_samples = 0
    
    with torch.no_grad():
        for images, features, _ in loader:
            images = images.to(device)
            if is_hybrid:
                # Pass both modalities for HAD-Net
                outputs = model(images, features.to(device))
            else:
                # Pass only images for Baseline
                outputs = model(images)
            
            center += torch.sum(outputs, dim=0)
            n_samples += outputs.shape[0]
            
    return center / n_samples

def train_one_epoch(model: torch.nn.Module, loader: DataLoader, center: torch.Tensor, 
                    optimizer: optim.Optimizer, device: torch.device, is_hybrid: bool = True) -> float:
    """
    Performs one training epoch by minimizing the distance between 
    outputs and the learned center of normality.
    
    Args:
        model: The model to train.
        loader: Training data loader.
        center: The target center in the latent space.
        optimizer: Adam or SGD optimizer.
        device: CPU or GPU.
        is_hybrid: Boolean to switch between HAD-Net and Baseline logic.
        
    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0
    
    for images, features, _ in loader:
        images = images.to(device)
        optimizer.zero_grad()
        
        if is_hybrid:
            outputs = model(images, features.to(device))
        else:
            outputs = model(images)
            
        # Deep SVDD Loss: Quadratic distance to the center
        # We aim to pull all healthy samples towards the center
        loss = torch.mean(torch.sum((outputs - center) ** 2, dim=1))
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)
