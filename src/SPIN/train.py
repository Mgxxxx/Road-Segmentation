import platform
import time
import torch
import os
from src.SPIN.model import SPINRoadMapperFCN8
from src.SPIN.dataloader import get_dataloaders
from src.config import DEVICE, EPOCHS, LEARNING_RATE, PATIENCE, BATCH_SIZE
import torch.optim as optim
import torch.nn as nn

if os.name == 'posix' and platform.system() == 'Darwin':
    torch.mps.empty_cache()

class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def check(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


from torch.nn import MSELoss

def train_model():
    # Data loaders
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # Model, optimizer, loss
    model = SPINRoadMapperFCN8().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    seg_criterion = nn.BCEWithLogitsLoss()
    orient_criterion = MSELoss()
    early_stopping = EarlyStopping(patience=PATIENCE)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_start_time = time.time()
        epoch_seg_loss = 0.0
        epoch_orient_loss = 0.0

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}] started...")

        # Batch-wise training
        for batch_idx, (images, masks) in enumerate(train_loader):
            batch_start_time = time.time()

            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()

            # Forward pass
            seg_output, orientation_output = model(images)

            # Compute segmentation loss
            seg_loss = seg_criterion(seg_output, masks)

            # Dummy target for orientation (you need actual orientation labels)
            orientation_target = torch.zeros_like(orientation_output).to(DEVICE)  # Placeholder
            orient_loss = orient_criterion(orientation_output, orientation_target)

            # Combined loss
            total_loss = seg_loss + 0.1 * orient_loss  # Weight orientation loss
            total_loss.backward()
            optimizer.step()

            # Track losses
            batch_time = time.time() - batch_start_time
            epoch_seg_loss += seg_loss.item()
            epoch_orient_loss += orient_loss.item()

            print(f"Batch {batch_idx + 1}/{len(train_loader)} | "
                  f"Seg Loss: {seg_loss.item():.4f} | "
                  f"Orient Loss: {orient_loss.item():.4f} | "
                  f"Time: {batch_time:.2f} sec")

        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_seg_loss = epoch_seg_loss / len(train_loader)
        avg_orient_loss = epoch_orient_loss / len(train_loader)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] completed | "
              f"Avg Seg Loss: {avg_seg_loss:.4f} | "
              f"Avg Orient Loss: {avg_orient_loss:.4f} | "
              f"Time: {epoch_time:.2f} sec")

        if early_stopping.check(avg_seg_loss):
            print("Early stopping triggered")
            break

    # Save the trained model
    checkpoint_dir = "results/current/checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")   