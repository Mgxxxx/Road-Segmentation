import torch
from src.model import SPINRoadMapper, SPINRoadMapperFCN8
import torchvision.models.segmentation as segmentation


DATA_PATH = "dataset/"
TRAIN_IMAGES_PATH = DATA_PATH + "training/images/"
TRAIN_MASKS_PATH = DATA_PATH + "training/groundtruth/"
TEST_IMAGES_PATH = DATA_PATH + "test_set_images/"

# Parameters
MODEL = SPINRoadMapperFCN8()
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 30
DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")