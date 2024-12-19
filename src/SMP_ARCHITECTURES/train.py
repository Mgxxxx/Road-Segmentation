import pytorch_lightning as pl
import torch

from src.SMP_ARCHITECTURES.model import RoadModel
from src.SMP_ARCHITECTURES.dataset import get_dataloaders
from src.config import EPOCHS, BATCH_SIZE, MODEL_PATH



def train(modelName):
    model_fpn = RoadModel("FPN", "resnet50", in_channels=3, out_classes=1)
    model_unet = RoadModel("Unet", "resnet50", in_channels=3, out_classes=1)

    model_unet_pretrained = RoadModel("Unet", "resnet50", in_channels=3, out_classes=1, encoder_weights="imagenet")
    model_unet_plus_pretrained = RoadModel("UnetPlusPlus", "resnet50", in_channels=3, out_classes=1, encoder_weights="imagenet")    
    
    if modelName == "FPN":
        model = model_fpn
    elif modelName == "UNET":
        model = model_unet
    elif modelName == "UNET_PRETRAINED":
        model = model_unet_pretrained
    else:
        raise ValueError("Invalid model name")
        
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    
    trainer = pl.Trainer(max_epochs=EPOCHS, log_every_n_steps=1)
    trainer.fit(
        model,
        train_loader,
        val_loader
    )
    
    torch.save(model.state_dict(), f"{MODEL_PATH}/SMP_model_{modelName}.pth")
    
        
    