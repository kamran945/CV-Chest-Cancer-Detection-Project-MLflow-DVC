import os
from pathlib import Path

import numpy as np

from PIL import Image
import torch
from torchvision import transforms


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    @staticmethod
    def load_model_checkpoint(path: Path) -> dict:
        return torch.load(path)

    def predict(self):

        checkpoint_model = self.load_model_checkpoint(
            os.path.join("artifacts", "training", "model.pth")
        )
        # checkpoint_model = torch.load(file_path)
        if "model" in checkpoint_model.keys():
            model = checkpoint_model["model"]
        else:
            model = None

        # Step 1: Define image transformations
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224)
                ),  # Resize image to the input size expected by the model
                transforms.ToTensor(),  # Convert image to PyTorch tensor
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Normalize using ImageNet standards
            ]
        )

        # Step 2: Load the image
        image = Image.open(self.filename).convert(
            "RGB"
        )  # Open image and convert to RGB

        # Step 3: Apply transformations
        image = transform(image)
        image = image.unsqueeze(0)  # Add a batch dimension

        # Step 4: Load model (e.g., a pre-trained ResNet)
        model.eval()  # Set model to evaluation mode

        # Step 5: Make predictions
        # Perform inference
        with torch.no_grad():  # Disable gradient calculation
            output = model(image)  # Forward pass
            probabilities = torch.softmax(
                output, dim=1
            )  # Apply softmax to get probabilities

        # Get the predicted class
        predicted_class = torch.argmax(
            probabilities, dim=1
        ).item()  # Class with highest probability
        # return predicted_class, probabilities

        # print(output)
        # print(probabilities)
        # print(predicted_class)

        if predicted_class == 0:
            prediction = "Tumor"
            return [{"image": prediction}]
        else:
            prediction = "Normal"
            return [{"image": prediction}]
        # return result


if __name__ == "__main__":
    try:
        # logger.info(f">>>> Starting {STAGE_NAME}. <<<<")
        pipeline = PredictionPipeline(
            "F://GitHub//CV-Chest-Cancer-Detection-Project-MLflow-DVC//artifacts//testing//000000 (6).png"
        )
        pipeline.predict()
        # logger.info(f">>>> {STAGE_NAME} completed successfully. <<<<")
    except Exception as e:
        # logger.error(f"Error occurred during {STAGE_NAME}: {str(e)}")
        raise e
