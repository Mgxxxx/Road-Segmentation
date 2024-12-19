
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

from src.UNET.unet import UNet
from src.UNET.dataset import get_dataloaders
from src.config import LEARNING_RATE, BATCH_SIZE, EPOCHS, MODEL_PATH, DEVICE, PATIENCE


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
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


def train_model():
    
    MODEL_SAVE_PATH = os.path.join(MODEL_PATH, "modelUnet.pth")
    
        
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE) 
                
                    
    model = UNet(in_channels=3, num_classes=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion  = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(patience=PATIENCE)
    
    for epoch in range(EPOCHS):
        model.train()
        train_running_loss = 0.0
        for idx, img_mask in enumerate(tqdm(train_loader)):
            img = img_mask[0].to(DEVICE)
            mask = img_mask[1].to(DEVICE)
            
            y_pred = model(img)
            optimizer.zero_grad()
            
            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        train_loss = train_running_loss / len(train_loader)
        
        model.eval()
        val_running_loss = 0.0
        
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_loader)):
                img = img_mask[0].to(DEVICE)
                mask = img_mask[1].to(DEVICE)
                
                y_pred = model(img)
                loss = criterion(y_pred, mask)
                
                val_running_loss += loss.item()
                
            val_loss = val_running_loss / len(val_loader)
            
        print(f"Epoch: {epoch+1}, Training Loss: {train_loss}, Validation Loss: {val_loss}")  
        
        if early_stopping.check(val_loss):
            print("Early stopping triggered")
            break      
            
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
