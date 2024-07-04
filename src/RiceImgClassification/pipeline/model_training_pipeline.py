from RiceImgClassification.config.configuration import ConfigureationManager
from RiceImgClassification.components.model_training import ModelTraining
from RiceImgClassification.components.data_preparation import DataPreparation
from RiceImgClassification import logger

STAGE_NAME = "MODEL TRAINING STAGE"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigureationManager()
        #data preparation
        data_preparation_config = config.get_data_preparation_config()
        data_preparation = DataPreparation(config=data_preparation_config)
        training_set, val_set = data_preparation.train_test_set()
        #model_training
        model_training_config = config.get_model_training_config()
        model_training = ModelTraining(config=model_training_config)
        cnn = model_training.get_model()
        trained_model, history = model_training.train(model=cnn, training_set=training_set, val_set=val_set)
        model_training.save_model(model=trained_model)
        model_training.save_history(history=history)
    

if __name__ == '__main__':
    try: 
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e