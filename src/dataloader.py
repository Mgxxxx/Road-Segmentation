import os
import glob
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from src.config import TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH, TEST_IMAGES_PATH

class RoadSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None):
        """
        Args:
            image_dir: Directory with images. Can contain subfolders.
            mask_dir: Directory with masks (only for training).
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # Recursively find all image files
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "**", "*.png"), recursive=True))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png"))) if mask_dir else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0  # Normalize to [0,1]

        if self.mask_dir:
            mask_path = self.mask_paths[idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (torch.tensor(mask).unsqueeze(0) > 0).float()  # Binary mask
            return image, mask

        return image

def get_dataloaders(batch_size):
    train_loader = DataLoader(
        RoadSegmentationDataset(TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH),
        batch_size=batch_size, shuffle=True
    )

    val_loader = DataLoader(
    RoadSegmentationDataset(TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH),  # Validation needs masks
    batch_size=batch_size, shuffle=False
    )
    
    test_loader = DataLoader(
        RoadSegmentationDataset(TEST_IMAGES_PATH),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader, val_loader