import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
import albumentations as A


from src.SMP_ARCHITECTURES.model import RoadModel
from src.SMP_ARCHITECTURES.dataset import get_dataloaders
from src.config import BATCH_SIZE, DEVICE, MODEL_PATH, PREDICTION_PATH


def show_validation_inferences(model_name):
    model = RoadModel("Unet", "resnet50", in_channels=3, out_classes=1)
    model_path = os.path.join(MODEL_PATH, f"SMP_model_{model_name}.pth")
    
    model.load_state_dict(torch.load(model_path))
    
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    
    batch = next(iter(val_loader))
    with torch.no_grad():
        model.eval()
        logits = model(batch[0])
    pr_masks = logits.sigmoid()
    pr_masks = (pr_masks > 0.5).float()
    for idx, (image, gt_mask, pr_mask) in enumerate(
        zip(batch[0], batch[1], pr_masks)
    ):
        if idx <= 9:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(image.numpy().transpose(1, 2, 0))
            plt.title("Image")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(gt_mask.numpy().squeeze())
            plt.title("Ground truth")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(pr_mask.numpy().squeeze())
            plt.title("Prediction")
            plt.axis("off")
            plt.show()
        else:
            break
        

def get_predictions(model_name):
    model = RoadModel("Unet", "resnet50", in_channels=3, out_classes=1)
    model_path = os.path.join(MODEL_PATH, f"SMP_model_{model_name}.pth")

    model.load_state_dict(torch.load(model_path, DEVICE), strict=False)
    
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=1)
    
    # Directory to save predictions
    predict_dir_path = os.path.join(PREDICTION_PATH, model_name)

    if not os.path.exists(predict_dir_path):
        os.makedirs(predict_dir_path)

    # Initialize index
    i = 0

    # Define transform for resizing
    resize_transform = transforms.Resize((608, 608))

    # Loop through test dataloader
    for image in tqdm(test_loader):
        with torch.no_grad():
            model.eval()
            logits = model(image)  # Forward pass to get predictions

        # Apply sigmoid activation and thresholding
        pr_masks = logits.sigmoid()
        pr_masks = (pr_masks > 0.5).float()

        # Extract the first mask and move to CPU
        pr_mask = pr_masks[0].squeeze(0).cpu().detach()
        
        # Resize mask to (608, 608)
        pr_mask = resize_transform(pr_mask.unsqueeze(0))  # Add channel for Resize
        pr_mask = pr_mask.squeeze(0)  # Remove extra dimension

        # Convert to PIL Image
        pr_mask_pil = transforms.ToPILImage()(pr_mask)

        # Save prediction
        i += 1
        pred_path = os.path.join(predict_dir_path, f"prediction_{i}.png")
        pr_mask_pil.save(pred_path)

    print(f"Predictions saved at {predict_dir_path}")