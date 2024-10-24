import os
from pathlib import Path
from zipfile import ZipFile


import urllib.request as request

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from src.lungCancerDetection.entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):

        # Load the pretrained MobileNetV2 model
        self.model = models.mobilenet_v2(pretrained=True)

        # Modify the classifier for binary classification
        n_features = self.model.classifier[
            1
        ].in_features  # Number of input features to the classifier
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(n_features, 1),  # Binary classification (one output node)
            nn.Sigmoid(),  # Sigmoid activation for binary classification
        )

        # Freeze the base layers
        for param in self.model.parameters():
            param.requires_grad = False

        checkpoint = {"model": self.model}
        self.save_model_checkpoint(
            path=self.config.base_model_path, checkpoint=checkpoint
        )

    def _prepare_model_for_classification(
        self, model, classes, freeze_all, freeze_till, pre_trained_lr, clf_lr
    ):
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False

            # Define optimizer with different learning rates for different parts of the model
            self.optimizer = optim.Adam(
                [
                    {
                        "params": model.classifier.parameters(),
                        "lr": clf_lr,
                    },  # Classifier (fine-tuned)
                ]
            )
        elif (freeze_till is not None) and (freeze_till > 0):
            for name, param in model.features[-freeze_till:].named_parameters():
                param.requires_grad = True

            # Define optimizer with different learning rates for different parts of the model
            self.optimizer = optim.Adam(
                [
                    {
                        "params": model.features.parameters(),
                        "lr": pre_trained_lr,
                    },  # Pretrained layers
                    {
                        "params": model.classifier.parameters(),
                        "lr": clf_lr,
                    },  # Classifier (fine-tuned)
                ]
            )

        return model, self.optimizer

    def update_base_model(self):
        self.full_model, self.optimizer = self._prepare_model_for_classification(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            pre_trained_lr=self.config.params_pre_trained_lr,
            clf_lr=self.config.params_clf_lr,
        )

        checkpoint = {"model": self.full_model, "optimizer": self.optimizer}

        self.save_model_checkpoint(
            path=self.config.updated_base_model_path,
            checkpoint=checkpoint,
        )

    @staticmethod
    def save_model_checkpoint(path: Path, checkpoint: dict) -> None:
        """
        Save a PyTorch model and its associated optimizer state dictionary.
        """
        # Save the checkpoint dictionary
        torch.save(checkpoint, path)
