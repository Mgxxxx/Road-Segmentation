import torch
DATA_PATH = "dataset/"
TRAIN_IMAGES_PATH = DATA_PATH + "training/images/"
TRAIN_MASKS_PATH = DATA_PATH + "training/groundtruth/"
TEST_IMAGES_PATH = DATA_PATH + "test_set_images/"

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 80
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")