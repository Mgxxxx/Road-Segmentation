from matplotlib.dates import MO
import torch
DATA_PATH = "data/"
TRAIN_IMAGES_PATH = DATA_PATH + "training/images/"
TRAIN_MASKS_PATH = DATA_PATH + "training/groundtruth/"
VAL_IMAGES_PATH = DATA_PATH + "validation/images/"
VAL_MASKS_PATH = DATA_PATH + "validation/groundtruth/"
TEST_IMAGES_PATH = DATA_PATH + "test_set_images/"

MODEL_PATH = "models/"

# Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
PATIENCE = 5
EPOCHS = 2
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")