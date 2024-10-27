from src.lungCancerDetection import logger

from src.lungCancerDetection.pipeline.stage_01_data_ingestion_pipeline import (
    DataIngestionTrainingPipeline,
)
from src.lungCancerDetection.pipeline.stage_02_prepare_base_model_pipeline import (
    PrepareBaseModelTrainingPipeline,
)
from src.lungCancerDetection.pipeline.stage_03_model_training_pipeline import (
    ModelTrainingPipeline,
)

from src.lungCancerDetection.pipeline.stage_04_model_evaluation_pipeline import (
    EvaluationPipeline,
)

STAGE_NAME = "DataIngestion Stage"

try:
    logger.info(f">>>> Starting {STAGE_NAME}. <<<<")
    pipeline = DataIngestionTrainingPipeline()
    pipeline.main()
    logger.info(f">>>> {STAGE_NAME} completed successfully. <<<<")
except Exception as e:
    logger.error(f"Error occurred during {STAGE_NAME}: {str(e)}")
    raise e


STAGE_NAME = "Prepare Base Model Stage"

try:
    logger.info(f">>>> Starting {STAGE_NAME}. <<<<")
    pipeline = PrepareBaseModelTrainingPipeline()
    pipeline.main()
    logger.info(f">>>> {STAGE_NAME} completed successfully. <<<<")
except Exception as e:
    logger.error(f"Error occurred during {STAGE_NAME}: {str(e)}")
    raise e


STAGE_NAME = "Model Training Stage"

try:
    logger.info(f">>>> Starting {STAGE_NAME}. <<<<")
    pipeline = ModelTrainingPipeline()
    pipeline.main()
    logger.info(f">>>> {STAGE_NAME} completed successfully. <<<<")
except Exception as e:
    logger.error(f"Error occurred during {STAGE_NAME}: {str(e)}")
    raise e


STAGE_NAME = "Model Evaluation Stage"

try:
    logger.info(f">>>> Starting {STAGE_NAME}. <<<<")
    pipeline = EvaluationPipeline()
    pipeline.main()
    logger.info(f">>>> {STAGE_NAME} completed successfully. <<<<")
except Exception as e:
    logger.error(f"Error occurred during {STAGE_NAME}: {str(e)}")
    raise e
