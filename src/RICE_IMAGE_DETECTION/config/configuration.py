from src.RICE_IMAGE_DETECTION.constants import *
from src.RICE_IMAGE_DETECTION.utils.common import read_yaml, create_directories
from src.RICE_IMAGE_DETECTION.entity.config_entity import DataIngestionConfig, DataPreparationConfig, ModelPreparationConfig
import os

class ConfigureationManager:
    def __init__(self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_cofig = DataIngestionConfig(
            root_dir = config.root_dir,
            source_url = config.source_url,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )
        return data_ingestion_cofig
    
    def get_data_preparation_config(self) -> DataPreparationConfig:
        config = self.config.data_preparation

        create_directories([config.root_dir])

        data_preparation_config = DataPreparationConfig(
            root_dir = config.root_dir,
            data_dir = config.data_dir,
            train_dir = config.train_dir,
            test_dir = config.test_dir
        )

        return data_preparation_config
    
    def get_model_preparation_config(self) -> ModelPreparationConfig:
        config = self.config.prepare_model

        create_directories([config.root_dir])

        model_preparation_config = ModelPreparationConfig(
            root_dir = Path(config.root_dir),
            model_dir = Path(config.model_dir),
            weight_decay = self.params.weight_decay,
            classes = self.params.classes,
            input_image_size= self.params.input_image_size
        )

        return model_preparation_config