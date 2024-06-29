import os
import gdown
import zipfile
from src.RICE_IMAGE_DETECTION import logger
from src.RICE_IMAGE_DETECTION.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_data(self)->str:
        try:
            data_url = self.config.source_url
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {data_url} into file {zip_download_dir}")

            file_id = data_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id,zip_download_dir)

            logger.info(f"Downloaded data from {data_url} into file {zip_download_dir}")
        except Exception as e:
            raise e
        
    def unzip_data(self):
        unzip_path = self.config.unzip_dir
        zip_download_dir = self.config.local_data_file
        os.makedirs(unzip_path, exist_ok=True)
        logger.info(f"unzipping data from {zip_download_dir} into file {unzip_path}")
        with zipfile.ZipFile(zip_download_dir, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"unzipping completed data from {zip_download_dir} into file {unzip_path}")