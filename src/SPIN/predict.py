from src.SPIN.model import SPINRoadMapperFCN8
from src.SPIN.dataloader import RoadSegmentationDataset
from src.config import DEVICE, TEST_IMAGES_PATH
from torch.utils.data import DataLoader
import torch
import os
import torchvision.transforms.functional as TF
from src.config import BATCH_SIZE


def generate_predictions():
    # Load model
    model = SPINRoadMapperFCN8().to(DEVICE)
    model.load_state_dict(torch.load("results/current/checkpoints/model.pth"))
    model.eval()

    # Load test data
    test_dataset = RoadSegmentationDataset(TEST_IMAGES_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create output folder
    output_folder = "results/current/predictions/"
    os.makedirs(output_folder, exist_ok=True)

    # Generate predictions
    print("Generating predictions...")
    for idx, images in enumerate(test_loader):
        images = images.to(DEVICE)
        with torch.no_grad():
            seg_output, _ = model(images)  # Unpack the outputs
            outputs = torch.sigmoid(seg_output)  # Apply sigmoid only to the segmentation output
        preds = (outputs > 0.5).float()  # Threshold predictions

        # Save predictions for the current batch
        for i, pred in enumerate(preds):
            # Extract the original filename
            image_index = idx * BATCH_SIZE + i
            if image_index >= len(test_dataset.image_paths):
                break  # Avoid out-of-bound errors
            image_name = os.path.basename(test_dataset.image_paths[image_index])
            image_name = os.path.splitext(image_name)[0]  # Remove file extension

            # Convert prediction to image
            pred_image = (pred.squeeze(0) * 255).byte()  # Convert to 8-bit
            output_path = os.path.join(output_folder, f"pred_{image_name}.png")
            TF.to_pil_image(pred_image).save(output_path)

    print("Predictions generated successfully!")