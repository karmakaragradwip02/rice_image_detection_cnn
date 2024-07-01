from src.RICE_IMAGE_DETECTION.config.configuration import ConfigureationManager
from src.RICE_IMAGE_DETECTION.components.model_training import ModelTraining
from src.RICE_IMAGE_DETECTION.components.data_preparation import DataPreparation
from src.RICE_IMAGE_DETECTION import logger

STAGE_NAME = "MODEL TRAINING STAGE"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigureationManager()
        #data preparation
        data_preparation_config = config.get_data_preparation_config()
        data_preparation = DataPreparation(config=data_preparation_config)
        training_set, test_set = data_preparation.train_test_set()
        #model_training
        model_training_config = config.get_model_training_config()
        model_training = ModelTraining(config=model_training_config)
        cnn = model_training.get_model()
        trained_model, history = model_training.train(model=cnn, training_set=training_set, test_set=test_set)
        model_training.save_model(model=trained_model)
        model_training.save_history(history=history)
        trained_model, history = model_training.all_modules(trained_model, history)
    

if __name__ == '__name__':
    try: 
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e