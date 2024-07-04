from RiceImgClassification import logger
import tensorflow as tf
from pathlib import Path
from RiceImgClassification.entity.config_entity import ModelPreparationConfig

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
    
    def model_compilation(self, model):
        #decay_rate = self.config.decay_rate
        #epsilon = self.config.epsilon
        learning_rate = self.config.learning_rate
        #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #initial_learning_rate=learning_rate, decay_steps=100000, decay_rate=decay_rate, staircase=True)
        #optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=epsilon)
        #model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def save_model(self, model):
        model_dir = Path(self.config.model_dir)
        model.save(model_dir)
        logger.info(f"---MODEL SAVED------")
