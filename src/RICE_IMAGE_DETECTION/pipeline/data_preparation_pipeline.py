from src.RICE_IMAGE_DETECTION.config.configuration import ConfigureationManager
from src.RICE_IMAGE_DETECTION.components.data_preparation import DataPreparation
from src.RICE_IMAGE_DETECTION import logger

STAGE_NAME = "DATA PREPARATION STAGE"

class DataPreparationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigureationManager()
        data_preparation_config = config.get_data_preparation_config()
        data_preparation = DataPreparation(config=data_preparation_config)
        data_preparation.making_traintest_folder()
        data_preparation.split_data()
        data_preparation.train_test_set()
    

if __name__ == '__name__':
    try: 
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreparationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e