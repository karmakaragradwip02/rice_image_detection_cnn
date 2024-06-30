from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen= True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen= True)
class DataPreparationConfig:
    root_dir: Path
    data_dir: Path
    train_dir: Path
    test_dir: Path

@dataclass(frozen= True)
class ModelPreparationConfig:
    root_dir: Path
    model_dir : Path
    weight_decay : float
    learning_rate : float
    epsilon : float
    batch_size : int
    decay_rate : float
    classes :int

@dataclass(frozen= True)
class ModelTrainingConfig:
    root_dir : Path
    trained_model_dir : Path
    training_data : Path
    epochs : int
    batch_size :int
    image_size : list

@dataclass(frozen= True)
class ModelEvaluation:
    trained_model_dir : Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    image_size: list
    batch_size: int