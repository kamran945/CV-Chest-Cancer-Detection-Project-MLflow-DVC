import os

import zipfile
import gdown

from lungCancerDetection.entity import DataIngestionConfig
from lungCancerDetection.utils.common import get_directory_size
from lungCancerDetection import logger


class DataIngestion:
    """
    Represents a data ingestion process.
    """

    def __init__(self, config: DataIngestionConfig) -> None:
        """
        Initialize the DataIngestion class with the given configuration.
        """
        self.config = config

    def download_data(self) -> None:
        """
        Download the data from the given source URL.
        """
        try:
            logger.info(f"Starting data download from: {self.config.source_URL}")
            gdrive_url = self.config.source_URL
            file_id = gdrive_url.split("/")[-2]
            url = f"https://drive.google.com/uc?/export=download&id={file_id}"
            output_path = self.config.local_data_file
            gdown.download(url, output_path, quiet=False)
            logger.info(f"Succesfully Downloaded data to: {output_path}")
        except Exception as e:
            logger.error(f"Error during data download: {str(e)}")

    def extract_data(self) -> None:
        """
        Extract the downloaded data to the specified local directory.
        """
        try:
            logger.info(f"Starting data extraction to: {self.config.unzip_dir}")
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)
            with zipfile.ZipFile(self.config.local_data_file, "r") as unzip:
                unzip.extractall(unzip_path)
                logger.info(f"Extracted data to: {unzip_path}")
                logger.info(
                    f"Total size of extracted data: {get_directory_size(unzip_path)} MB"
                )
        except Exception as e:
            logger.error(f"Error during data extraction: {str(e)}")
