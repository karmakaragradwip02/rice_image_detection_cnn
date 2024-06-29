from src.RICE_IMAGE_DETECTION import logger
from src.RICE_IMAGE_DETECTION.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.RICE_IMAGE_DETECTION.config.configuration import ConfigureationManager

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