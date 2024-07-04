import os
import json
import mlflow
import numpy as np
import mlflow.keras
import tensorflow as tf
from pathlib import Path
from urllib.parse import urlparse
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, cohen_kappa_score
from RICE_IMAGE_DETECTION.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def load_model(self):
        return tf.keras.models.load_model(self.config.trained_model_dir)
    
    def val_set(self):
        val_folder = self.config.val_dir
        val_datagen = ImageDataGenerator(rescale=1./255)
        val_set = val_datagen.flow_from_directory(
            val_folder,
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical',
            shuffle=False)
        return val_set
    
    def plot(self):
        history_path = Path(self.config.history_dir)
        # Load the data
        with open(history_path, 'r') as file:
            data = json.load(file)

        # Define the number of epochs
        epochs = range(1, self.config.epochs+1)

        # Apply a style
        plt.style.use('seaborn-v0_8-darkgrid')

        # Create a figure
        plt.figure(figsize=(7, 3.5))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, data["loss"], marker='o', linestyle='-', color='b', label='Training Loss')
        plt.plot(epochs, data["val_loss"], marker='o', linestyle='--', color='r', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss vs. Epochs')
        plt.legend(loc='best')
        plt.grid(True)

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, data["accuracy"], marker='o', linestyle='-', color='b', label='Training Accuracy')
        plt.plot(epochs, data["val_accuracy"], marker='o', linestyle='--', color='r', label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Epochs')
        plt.legend(loc='best')
        plt.grid(True)

        # Add some space between plots
        plt.tight_layout()
        graph_dir = self.config.graph_dir
        plt.savefig(graph_dir)
        # Show the plot
        plt.show()

    def log_into_mlflow(self, model, val_set):
        MLFLOW_TRACKING_URI = "https://dagshub.com/karmakaragradwip02/rice_image_detection_cnn.mlflow"
        os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
        os.environ['MLFLOW_TRACKING_USERNAME'] = 'karmakaragradwip02'
        os.environ['MLFLOW_TRACKING_PASSWORD'] = '9ccb0f28354fcca6469017b32544fa0704b9c343'

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        # Read and parse the history data
        history_path = Path(self.config.history_dir)
        if history_path.is_file():
            with history_path.open('r') as f:
                history_data = json.load(f)
                
            # Log each epoch's metrics individually
            with mlflow.start_run():
                mlflow.log_params(self.config.all_params)
                y_pred = np.argmax(model.predict(val_set), axis=1)
                y_true = val_set.classes

                # Calculate precision and recall
                precision = precision_score(y_true, y_pred, average='macro')
                recall = recall_score(y_true, y_pred, average='macro')
                m_accuracy = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='macro')
                kappa = cohen_kappa_score(y_true, y_pred)
                
                # Log metrics
                mlflow.log_metric('Model Accuracy', m_accuracy)
                mlflow.log_metric('Model Precision', precision)
                mlflow.log_metric('Model Recall', recall)
                mlflow.log_metric('Model F1 Score', f1)
                mlflow.log_metric('Model Kappa', kappa)

                print('Model Accuracy', m_accuracy)
                print('Model Precision', precision)
                print("recall", recall)
                print('Model Recall', recall)
                print('Model Kappa', kappa)

                for epoch, (loss, accuracy, val_loss, val_accuracy) in enumerate(zip(
                        history_data["loss"],
                        history_data["accuracy"],
                        history_data["val_loss"],
                        history_data["val_accuracy"])):
                    mlflow.log_metric("loss", loss, step=epoch)
                    mlflow.log_metric("accuracy", accuracy, step=epoch)
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                    mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
                if tracking_url_type_store != "file":
                    mlflow.keras.log_model(model, "model", registered_model_name="custom_model")
                else:
                    mlflow.keras.log_model(model, "model")