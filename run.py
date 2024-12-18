from src.train import train_model
from src.predict import generate_predictions

if __name__ == "__main__":
    # Train the model
    train_model()
    # Generate predictions for test images
    generate_predictions()