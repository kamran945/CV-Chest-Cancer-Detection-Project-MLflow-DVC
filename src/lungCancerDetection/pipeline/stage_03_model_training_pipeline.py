from src.lungCancerDetection.config.configuration import ConfigurationManager
from src.lungCancerDetection.components.model_training import Training
from src.lungCancerDetection import logger


STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_val_data_generators()
        training.train_model()


if __name__ == "__main__":
    try:
        logger.info(f">>>> Starting {STAGE_NAME}. <<<<")
        pipeline = ModelTrainingPipeline()
        pipeline.main()
        logger.info(f">>>> {STAGE_NAME} completed successfully. <<<<")
    except Exception as e:
        logger.error(f"Error occurred during {STAGE_NAME}: {str(e)}")
        raise e
