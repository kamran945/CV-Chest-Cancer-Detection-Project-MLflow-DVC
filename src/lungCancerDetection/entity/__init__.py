from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    unzip_dir_full: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: tuple
    params_pre_trained_lr: float
    params_clf_lr: float
    params_requires_grad: bool
    params_classes: int


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: str
    val_data: str
    params_epochs: int
    params_batch_size: int
    params_augmentation: bool
    params_image_size: tuple
    params_num_workers: int


@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    all_params: dict
    mlflow_uri: str
    val_data: Path
    params_epochs: int
    params_batch_size: int
    params_augmentation: bool
    params_image_size: tuple
    params_num_workers: int
    all_params: dict
