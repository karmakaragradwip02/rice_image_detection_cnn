import json
import tensorflow as tf
from RICE_IMAGE_DETECTION import logger
from RICE_IMAGE_DETECTION.entity.config_entity import ModelTrainingConfig

class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    def get_model(self):
        logger.info("-----------Model loaded from the model path----------")
        model_dir = self.config.model_dir
        cnn = tf.keras.models.load_model(model_dir)
        return cnn
    
    def train(self, model, training_set, val_set):
        logger.info("-----------Model training is beginning----------")
        epochs = self.config.epochs
        history = model.fit(training_set, epochs=epochs, validation_data=val_set)
        logger.info("-----------Model training is ending----------")
        return model, history
    
    def save_model(self, model):
        trained_model_dir = self.config.trained_model_dir
        model.save(trained_model_dir)
        logger.info("--------Model saved successfully--------")

    def save_history(self, history):
        history_dir = self.config.history_dir
        history_dict = history.history
        
        with open(history_dir, 'w') as f:
            json.dump(history_dict, f)
        
        logger.info(f"------History saved successfully at {history_dir}---------")