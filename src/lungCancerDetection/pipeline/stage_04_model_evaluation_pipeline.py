from src.lungCancerDetection.config.configuration import ConfigurationManager
from src.lungCancerDetection.components.model_evaluation_with_mlflow import Evaluation
from src.lungCancerDetection import logger


STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        evaluation.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info(f">>>> Starting {STAGE_NAME}. <<<<")
        pipeline = EvaluationPipeline()
        pipeline.main()
        logger.info(f">>>> {STAGE_NAME} completed successfully. <<<<")
    except Exception as e:
        logger.error(f"Error occurred during {STAGE_NAME}: {str(e)}")
        raise e
