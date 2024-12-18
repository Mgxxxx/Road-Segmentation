from src.SPIN.train import train_model as train_model_SPIN
from src.SPIN.predict import generate_predictions as generate_predictions_SPIN

if __name__ == "__main__":
    model = 'SPIN'
    
    if model == 'SPIN':
        # Train the model
        train_model_SPIN()()
        # Generate predictions for test images
        generate_predictions_SPIN()()