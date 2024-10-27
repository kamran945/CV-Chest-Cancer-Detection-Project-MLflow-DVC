import os
import urllib.request as request
from zipfile import ZipFile
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.lungCancerDetection.entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_base_model(self):
        model_checkpoint = torch.load(
            self.config.updated_base_model_path,
            weights_only=False,
        )

        if "model" in model_checkpoint.keys():
            self.model = model_checkpoint["model"]
            # Move model to GPU if available
            self.model = self.model.to(self.device)
        else:
            self.model = None

        if "optimizer" in model_checkpoint.keys():
            self.optimizer = model_checkpoint["optimizer"]
        else:
            self.optimizer = None

    def train_val_data_generators(self):

        if self.config.params_augmentation:
            # Define image transformations
            train_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(20),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

            # val_transform = transforms.Compose(
            #     [
            #         transforms.Resize((224, 224)),
            #         transforms.ToTensor(),
            #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            #     ]
            # )
            val_transform = transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            )
        else:
            train_transform = transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            )
            val_transform = transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            )
        # Load datasets
        train_dataset = datasets.ImageFolder(
            root=self.config.training_data, transform=train_transform
        )
        val_dataset = datasets.ImageFolder(
            root=self.config.val_data, transform=val_transform
        )
        print(f"dataset classes: {print(train_dataset.class_to_idx)}")

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=True,
            # num_workers=self.config.params_num_workers,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False,
            # num_workers=self.config.params_num_workers,
        )

    @staticmethod
    def save_model_checkpoint(path: Path, model_checkpoint: dict):
        torch.save(model_checkpoint, path)

    def train_model(self):
        # Define loss function
        criterion = torch.nn.BCELoss()  # Binary cross-entropy loss
        for epoch in range(self.config.params_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.float().unsqueeze(
                    1
                )  # Make sure labels are float (required for BCELoss)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                preds = (
                    outputs > 0.5
                ).float()  # Sigmoid threshold of 0.5 for binary classification
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            # Calculate training loss and accuracy
            epoch_loss = running_loss / total
            epoch_acc = correct / total

            print(
                f"Epoch {epoch+1}/{self.config.params_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}"
            )

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    labels = labels.float().unsqueeze(1)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    preds = (outputs > 0.5).float()
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_loss = val_loss / total
            val_acc = correct / total
            print(
                f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}"
            )

        self.save_model_checkpoint(
            path=self.config.trained_model_path,
            model_checkpoint={"model": self.model, "optimizer": self.optimizer},
        )
