from RiceImgClassification.constants import *
from RiceImgClassification.utils.common import read_yaml, create_directories
from RiceImgClassification.entity.config_entity import DataIngestionConfig, DataPreparationConfig, ModelPreparationConfig, ModelTrainingConfig, ModelEvaluationConfig

class ConfigurationManager:
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
            test_dir = config.test_dir,
            val_dir = config.val_dir
        )

        return data_preparation_config
    
    def get_model_preparation_config(self) -> ModelPreparationConfig:
        config = self.config.prepare_model

        create_directories([config.root_dir])

        model_preparation_config = ModelPreparationConfig(
            root_dir = Path(config.root_dir),
            model_dir = Path(config.model_dir),
            weight_decay = self.params.weight_decay,
            input_image_size= self.params.input_image_size,
            learning_rate = self.params.learning_rate,
            epsilon = self.params.epsilon,
            classes = self.params.classes,
            epochs = self.params.epochs,
            decay_rate = self.params.decay_rate
        )

        return model_preparation_config
    
    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.training

        create_directories([config.root_dir])

        model_training_config = ModelTrainingConfig(
            root_dir = Path(config.root_dir),
            model_dir = Path(config.model_dir),
            trained_model_dir = Path(config.trained_model_dir),
            history_dir= Path(config.history_dir),
            epochs = self.params.epochs
        )

        return model_training_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir = Path(config.root_dir),
            trained_model_dir = Path(config.trained_model_dir),
            history_dir= Path(config.history_dir),
            graph_dir = Path(config.graph_dir),
            val_dir= Path(config.val_dir),
            mlflow_uri="https://dagshub.com/karmakaragradwip02/rice_image_detection_cnn.mlflow",
            all_params=self.params,
            epochs = self.params.epochs
        )

        return model_evaluation_config