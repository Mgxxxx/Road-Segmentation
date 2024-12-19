from src.UNET.train import train_model as train_model_UNET
from src.UNET.train import train_model as train_model_UNET
from src.SPIN.train import train_model as train_model_SPIN
from src.SPIN.predict import generate_predictions as generate_predictions_SPIN
from src.UNET.inference import predict_show_image, test_predictions
from src.SMP_ARCHITECTURES.train import train as train_model_SMP
from src.SMP_ARCHITECTURES.inference import show_validation_inferences as show_validation_inferences_SMP, get_predictions as get_predictions_SMP
from src.config import MODEL_PATH
import os

if __name__ == "__main__":
    model = 'SMP'
    
    if model == 'SPIN':
        # Train the model
        # train_model_SPIN()
        # Generate predictions for test images
        generate_predictions_SPIN()
    elif model == 'UNET':
        
        # train_model_UNET()
        
        model = os.path.join(MODEL_PATH, 'modelUnet.pth')
        predict_show_image(model)
        # test_predictions(model)
    elif model == 'SMP':
        model_name = 'UNET_PRETRAINED'
    
        #train_model_SMP(model_name)
        #show_validation_inferences_SMP(model_name)
        get_predictions_SMP(model_name)
    else:
        print("Invalid model name")
        
        