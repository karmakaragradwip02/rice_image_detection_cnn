from src.RICE_IMAGE_DETECTION.config.configuration import ConfigureationManager
from src.RICE_IMAGE_DETECTION.components.model_preparation import ModelPreparation
from src.RICE_IMAGE_DETECTION import logger

STAGE_NAME = "MODEL PREPARATION STAGE"

class ModelPreparationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigureationManager()
        model_preparation_config = config.get_model_preparation_config()
        model_preparation = ModelPreparation(config=model_preparation_config)
        cnn = model_preparation.model()
        model = model_preparation.model_compilation(cnn)
        model_preparation.save_model(model)
        logger.info(f"---MODEL SUMMARY------")
        logger.info(cnn.summary())
    

if __name__ == '__name__':
    try: 
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelPreparationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e