from src.RICE_IMAGE_DETECTION import logger
from src.RICE_IMAGE_DETECTION.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.RICE_IMAGE_DETECTION.pipeline.data_preparation_pipeline import DataPreparationTrainingPipeline
from src.RICE_IMAGE_DETECTION.pipeline.model_preparation_pipeline import ModelPreparationTrainingPipeline
from src.RICE_IMAGE_DETECTION.pipeline.model_training_pipeline import ModelTrainerTrainingPipeline
from src.RICE_IMAGE_DETECTION.pipeline.model_evaluation_pipeline import ModelEvaluationTrainingPipeline

STAGE_NAME = "DATA INGESTION STAGE"

try:
    logger.info(f"********************") 
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    logger.info(f"********************")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "DATA PREPARATION STAGE"

try:
    logger.info(f"********************") 
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataPreparationTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    logger.info(f"********************")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "MODEL PREPARATION STAGE"

try:
    logger.info(f"********************") 
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelPreparationTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    logger.info(f"********************") 
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "MODEL TRAINING STAGE"

try: 
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainerTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "MODEL EVALUATION STAGE"

try: 
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelEvaluationTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)
    raise e