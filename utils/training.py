import torch
import torch.optim as optim

def init_center(model, loader, device, is_hybrid=True):
    model.eval()
    center = torch.zeros(128).to(device)
    n_samples = 0
    with torch.no_grad():
        for images, features, _ in loader:
            images = images.to(device)
            if is_hybrid:
                outputs = model(images, features.to(device))
            else:
                outputs = model(images)
            center += torch.sum(outputs, dim=0)
            n_samples += outputs.shape[0]
    return center / n_samples

def train_one_epoch(model, loader, center, optimizer, device, is_hybrid=True):
    model.train()
    total_loss = 0
    for images, features, _ in loader:
        images = images.to(device)
        optimizer.zero_grad()
        
        if is_hybrid:
            outputs = model(images, features.to(device))
        else:
            outputs = model(images)
            
        loss = torch.mean(torch.sum((outputs - center) ** 2, dim=1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
