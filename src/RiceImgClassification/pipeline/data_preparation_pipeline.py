from RiceImgClassification.config.configuration import ConfigurationManager
from RiceImgClassification.components.data_preparation import DataPreparation
from RiceImgClassification import logger

STAGE_NAME = "DATA PREPARATION STAGE"

class DataPreparationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preparation_config = config.get_data_preparation_config()
        data_preparation = DataPreparation(config=data_preparation_config)
        data_preparation.making_traintest_folder()
        data_preparation.split_data()
        training_set, test_set = data_preparation.train_test_set()
    

if __name__ == '__main__':
    try: 
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreparationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e