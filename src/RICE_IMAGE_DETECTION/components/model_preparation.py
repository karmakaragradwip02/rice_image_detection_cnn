from src.RICE_IMAGE_DETECTION import logger
import tensorflow as tf
from pathlib import Path
from src.RICE_IMAGE_DETECTION.entity.config_entity import ModelPreparationConfig

class ModelPreparation:
    def __init__(self, config: ModelPreparationConfig):
        self.config = config

    def model(self):
        logger.info(f"------------Model Preparation Started-------------")
        classes = self.config.classes
        weight_decay = self.config.weight_decay
        input_shape = self.config.input_image_size
        cnn = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape,
                                kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=128, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
            tf.keras.layers.Dense(classes, activation='softmax')
            ])
        logger.info(f"------------Model Preparation Ended-------------")
        return cnn
    
    def save_model(self, model):
        model_dir = Path(self.config.model_dir)
        model.save(model_dir)
        logger.info(f"---MODEL SAVED------")
