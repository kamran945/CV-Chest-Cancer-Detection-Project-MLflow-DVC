from src.lungCancerDetection.config.configuration import ConfigurationManager
from src.lungCancerDetection.components.data_ingestion import DataIngestion
from src.lungCancerDetection import logger


STAGE_NAME = "DataIngestion Stage"


class DataIngestionTrainingPipeline:

    def __init__(self) -> None:
        """Initialize the pipeline"""
        pass

    def main(self) -> None:
        """Execute the pipeline"""

        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()

        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_data()
        data_ingestion.extract_data()


if __name__ == "__main__":
    try:
        logger.info(f">>>> Starting {STAGE_NAME}. <<<<")
        pipeline = DataIngestionTrainingPipeline()
        pipeline.main()
        logger.info(f">>>> {STAGE_NAME} completed successfully. <<<<")
    except Exception as e:
        logger.error(f"Error occurred during {STAGE_NAME}: {str(e)}")
        raise e
