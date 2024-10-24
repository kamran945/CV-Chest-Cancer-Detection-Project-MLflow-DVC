from pathlib import Path
import os

from src.lungCancerDetection.constants import *
from src.lungCancerDetection.utils.common import read_yaml_file, create_directories
from src.lungCancerDetection.entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
    EvaluationConfig,
)


class ConfigurationManager:
    """
    Class to manage the configuration parameters and initialize configurations.
    """

    def __init__(
        self,
        config_file_path: Path = CONFIG_FILE_PATH,
        params_file_path: Path = PARAMS_FILE_PATH,
    ) -> None:
        """
        Initialize the ConfigurationManager with the provided file paths.
        """

        self.config = read_yaml_file(config_file_path)
        self.params = read_yaml_file(params_file_path)

        create_directories(filepath_list=[self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Return a DataIngestionConfig object initialized with the configuration parameters.
        """

        config = self.config.data_ingestion

        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
            unzip_dir_full=config.unzip_dir_full,
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_pre_trained_lr=self.params.PRE_TRAINED_LR,
            params_clf_lr=self.params.CLF_LR,
            params_requires_grad=self.params.REQUIRES_GRAD,
            params_classes=self.params.CLASSES,
        )

        return prepare_base_model_config

    def get_training_config(self) -> TrainingConfig:

        training = self.config.training
        prepare_base_model = self.config.prepare_base_model

        params = self.params

        training_data = Path(self.config.data_ingestion.unzip_dir_full) / "train"
        val_data = Path(self.config.data_ingestion.unzip_dir_full) / "valid"

        create_directories([training.root_dir])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=training_data,
            val_data=val_data,
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_num_workers=params.NUM_WORKERS,
        )

        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:

        config = self.config.evaluation
        params = self.params

        val_data = Path(self.config.data_ingestion.unzip_dir_full) / "valid"

        eval_config = EvaluationConfig(
            path_of_model=config.path_of_model,
            mlflow_uri=config.mlflow_uri,
            val_data=val_data,
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_num_workers=params.NUM_WORKERS,
            all_params=params,
        )

        return eval_config
