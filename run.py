from src.UNET.train import train_model as train_model_UNET
from src.SPIN.train import train_model as train_model_SPIN
from src.SPIN.predict import generate_predictions as generate_predictions_SPIN
from src.UNET.inference import predict_show_image, test_predictions
from src.config import MODEL_PATH
import os

if __name__ == "__main__":
    model = 'UNET'
    
    if model == 'SPIN':
        # Train the model
        train_model_SPIN()
        # Generate predictions for test images
        generate_predictions_SPIN()
    elif model == 'UNET':
        
        train_model_UNET()
        
        model = os.path.join(MODEL_PATH, 'modelUnet.pth')
        predict_show_image(model)
        # test_predictions(model)
        
        