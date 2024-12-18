import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from src.config import TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH, VAL_IMAGES_PATH, VAL_MASKS_PATH, TEST_IMAGES_PATH
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch


class RoadDataset(Dataset):
    def __init__(self, image_path, mask_path, transforms=None):
        self.img_path = image_path
        self.mask_path = mask_path
        self.images = sorted([os.path.join(image_path, x) for x in os.listdir(image_path)])
        self.masks = sorted([os.path.join(mask_path, x) for x in os.listdir(mask_path)])
        if transforms:
            self.transforms = transforms
        else:
            self.transforms = A.Compose([
                A.Resize(512, 512),
                ToTensorV2()
            ])
        
    def __getitem__(self, index):
        image = np.array(Image.open(self.images[index]).convert("RGB"))
        image = (image / 255.0).astype(np.float32)
        mask = np.array(Image.open(self.masks[index]))
        mask = (mask > 0).astype(np.float32)
        
        transformed = self.transforms(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]
        
        mask = torch.unsqueeze(mask, 0)
        
        return image, torch.tensor(mask, dtype=torch.float32)
    
    def __len__(self):
        return len(self.images)
    
    
def list_all_files(rootdir):
    files = []
    for dirs in os.listdir(rootdir):
        for file in os.listdir(os.path.join(rootdir, dirs)):
            files.append(os.path.join(rootdir, dirs, file))
            
    files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
    return files
        


class testDataset(Dataset):
    def __init__(self, input_dir):
        self.images = list_all_files(input_dir)
        self.transform = A.Compose([
            A.Resize(512, 512),
            ToTensorV2()
        ])
        
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]).convert("RGB"))
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
            
                        
        return image
    
def get_dataloaders(batch_size, augmentations=None):
    train_loader = torch.utils.data.DataLoader(
        RoadDataset(TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH, augmentations),
        batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        RoadDataset(VAL_IMAGES_PATH, VAL_MASKS_PATH),
        batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        testDataset(TEST_IMAGES_PATH),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader, test_loader
    
    