from pathlib import Path
from urllib.parse import urlparse


import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import mlflow
import mlflow.pytorch

from src.lungCancerDetection.entity import EvaluationConfig
from src.lungCancerDetection.utils.common import (
    read_yaml,
    create_directories,
    save_json,
)


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):

        val_dataset = datasets.ImageFolder(root=self.config.val_data)

        # Create data loaders
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False,
            NUM_WORKERS=self.config.params_num_workers,
        )

    @staticmethod
    def load_model_checkpoint(path: Path) -> dict:
        return torch.load(path)

    def evaluation(self):
        model_checkpoint = self.load_model_checkpoint(self.config.path_of_model)

        if "model" in model_checkpoint.keys():
            self.model = model_checkpoint["model"]
        else:
            self.model = None

        if "optimizer" in model_checkpoint.keys():
            self.optimizer = model_checkpoint["optimizer"]
        else:
            self.optimizer = None

        self._valid_generator()

        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Define loss function
        criterion = torch.nn.BCELoss()  # Binary cross-entropy loss

        # Validation phase
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.float().unsqueeze(1)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        self.val_loss = val_loss / total
        self.val_acc = correct / total
        print(
            f"Validation Loss: {self.val_loss:.4f}, Validation Accuracy: {self.val_acc:.4f}"
        )

        self.save_score()

    def save_score(self):
        scores = {"loss": self.val_loss, "accuracy": self.val_acc}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.val_loss, "accuracy": self.val_acc})

            if tracking_url_type_store != "file":
                mlflow.pytorch.log_model(
                    self.model, "model", registered_model_name="MobileNetV2"
                )
            else:
                mlflow.pytorch.log_model(self.model, "model")
