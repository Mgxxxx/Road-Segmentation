import time
import torch
import os
from src.dataloader import get_dataloaders
from src.model import SPINRoadMapper
from src.config import DEVICE, EPOCHS, LEARNING_RATE, BATCH_SIZE, MODEL
import torch.optim as optim
import torch.nn as nn


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def check(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def validate_model(model, val_loader, criterion):
    """
    Runs validation on the validation dataset and computes average loss.
    """
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0

    with torch.no_grad():  # Disable gradient computation
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    return val_loss / len(val_loader)


def train_model():
    # Data loaders
    train_loader, _, val_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # Model, optimizer, loss, scheduler, and early stopping
    model = MODEL.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(patience=7, min_delta=0.001)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_start_time = time.time()
        epoch_loss = 0.0

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}] started...")

        # Batch-wise training
        for batch_idx, (images, masks) in enumerate(train_loader):
            batch_start_time = time.time()

            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            # Calculate time and loss
            batch_time = time.time() - batch_start_time
            epoch_loss += loss.item()

            print(f"Batch {batch_idx + 1}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Time: {batch_time:.2f} sec")

        # Compute average training loss
        avg_loss = epoch_loss / len(train_loader)

        # Run validation and compute validation loss
        val_loss = validate_model(model, val_loader, criterion)

        # Update the learning rate scheduler
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch [{epoch + 1}/{EPOCHS}] completed | "
              f"Avg Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.2f} sec")

        # Check early stopping
        if early_stopping.check(val_loss):
            print(f"Early stopping triggered at epoch {epoch + 1}. Best Val Loss: {early_stopping.best_loss:.4f}")
            break

    # Save the trained model
    checkpoint_dir = "results/current/checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure directory exists
    checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")    