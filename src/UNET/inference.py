import random
import test
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os

from tqdm import tqdm

from src.UNET.unet import UNet
from src.UNET.dataset import get_dataloaders, testDataset
from src.config import DEVICE, TEST_IMAGES_PATH, PREDICTION_PATH

def predict_show_image(model_pth):
    model = UNet(in_channels=3, num_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(model_pth, DEVICE))
    
    
    test_dataset = testDataset(TEST_IMAGES_PATH)
    
    idx = random.randint(0, len(test_dataset)-1)

    img = test_dataset[idx]
                    
    img = img.float().to(DEVICE)
    img = img.unsqueeze(0)
    
    pred_mask = model(img)
    
    img = img.squeeze(0).cpu().detach()
    img = img.permute(1, 2, 0)
        
    pred_mask = pred_mask.squeeze(0).cpu().detach()
    pred_mask = pred_mask.permute(1, 2, 0)
            
    pred_mask[pred_mask > 0] = 1
    pred_mask[pred_mask < 0] = 0
    
            
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img / 255.0)
    ax.set_title('Image')
    
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(pred_mask)
    ax.set_title('Predicted Mask')
    
    plt.show()
    
    
def test_predictions(model_pth):
    model = UNet(in_channels=3, num_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(model_pth, DEVICE))

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=1)
    
    images = []
    pred_masks = []
    
    i = 0
    
    predict_dir_path = os.path.join(PREDICTION_PATH, 'UNET')
    if not os.path.exists(predict_dir_path):
        os.makedirs(predict_dir_path)
    
    for img in tqdm(test_loader):
        i += 1
        img = img.float().to(DEVICE)
        img = img.unsqueeze(0)
        
        pred_mask = model(img)
        
        img = img.squeeze(0).cpu().detach()
        img = img.permute(1, 2, 0)
        
        pred_mask = pred_mask.squeeze(0).cpu().detach()
                
        pred_mask[pred_mask > 0] = 1
        pred_mask[pred_mask < 0] = 0
        
        pred_mask = transforms.Resize(size=(608, 608))(pred_mask)
        
        pred_path = os.path.join(predict_dir_path, f'test_mask_{i}.png')
        
        pred_mask = transforms.ToPILImage()(pred_mask)
        pred_mask.save(pred_path)
        
    print(f"Predictions saved at {predict_dir_path}")