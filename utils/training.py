import torch
import torch.optim as optim
import matplotlib.pyplot as plt

def init_center(model, loader, device, latent_dim=128):
    """Calculates the center 'c' of the hypersphere (Deep SVDD)."""
    print("Initializing Hypersphere Center...")
    model.eval()
    center = torch.zeros(latent_dim).to(device)
    n_samples = 0
    with torch.no_grad():
        for images, features, _ in loader:
            images, features = images.to(device), features.to(device)
            outputs = model(images, features)
            n_samples += outputs.shape[0]
            center += torch.sum(outputs, dim=0)
    center = center / n_samples
    # Prevent collapse
    center[(center.abs() < 0.1) & (center > 0)] = 0.1
    center[(center.abs() < 0.1) & (center < 0)] = -0.1
    return center

def train_model(model, train_loader, center, device, epochs=20, lr=0.001):
    """Main training loop using One-Class Center Loss."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    train_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, features, _ in train_loader:
            images, features = images.to(device), features.to(device)
            optimizer.zero_grad()
            outputs = model(images, features)
            # Loss: Minimize distance to center
            dist = torch.sum((outputs - center) ** 2, dim=1)
            loss = torch.mean(dist)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")
    
    return train_losses
