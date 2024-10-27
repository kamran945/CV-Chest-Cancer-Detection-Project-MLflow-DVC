from pathlib import Path
import os
import json

from ensure import ensure_annotations
from box import ConfigBox
from box.exceptions import BoxValueError
import yaml

from src.lungCancerDetection import logger


@ensure_annotations
def read_yaml_file(filename: Path) -> ConfigBox:
    """
    Read a YAML file and return its contents as a ConfigBox.
    Args:
        filename (Path): Path to the YAML file.
    Raises:
        BoxValueError: If the YAML file is empty or malformed.
        Exception: If an error occurs while loading the YAML file.
    Returns:
        ConfigBox: Contents of the YAML file.
    """
    try:
        with open(filename, "r") as file:
            config_data = yaml.safe_load(file)
            logger.info(f"YAML file {filename} loaded successfully!")
            return ConfigBox(config_data)
    except BoxValueError as e:
        raise ValueError("YAML file is empty or malformed. ", e)
    except Exception as e:
        raise Exception(f"An error occurred while loading the YAML file: {str(e)}")


@ensure_annotations
def create_directories(filepath_list: list):
    """
    Create directories based on the provided file paths.
    Args:
        filepath_list (list): List of file paths.
    Raises:
        Exception: If an error occurs while creating directories.
    """
    for filepath in filepath_list:
        try:
            if not os.path.exists(filepath):
                os.makedirs(filepath, exist_ok=True)
                logger.info(f"Directory {filepath} created successfully!")
            else:
                logger.info(f"Directory {filepath} already exists!")
        except Exception as e:
            raise Exception(f"An error occurred while creating directory: {str(e)}")


@ensure_annotations
def get_directory_size(dir_path: str) -> float:
    """
    Get the total size of all files in a directory and its subdirectories in MB.
    Args:
        dir_path (str): Path to the directory.
    Returns:
        int: Total size of files in MB.
    """
    total_size = 0
    for dirpath, _, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
    return total_size / (1024 * 1024)


# @ensure_annotations
def save_json(filepath: Path, data: dict) -> None:
    """
    Save data as a JSON file.
    Args:
        filepath (Path): Path to the JSON file.
        data (dict): Data to be saved.
    Raises:
        Exception: If an error occurs while saving the JSON file.
    """
    try:
        with open(filepath, "w") as file:
            json.dump(data, file, indent=4)
        logger.info(f"JSON file {filepath} saved successfully!")
    except Exception as e:
        raise Exception(f"An error occurred while saving JSON file: {str(e)}")


@ensure_annotations
def load_json(filepath: Path) -> ConfigBox:
    """
    Load a JSON file and return its contents as a ConfigBox.
    Args:
        filepath (Path): Path to the JSON file.
    Raises:
        BoxValueError: If the JSON file is empty or malformed.
        Exception: If an error occurs while loading the JSON file.
    Returns:
        ConfigBox: Contents of the JSON file.
    """
    try:
        with open(filepath, "r") as file:
            data = json.load(file)
            logger.info(f"JSON file {filepath} loaded successfully!")
            return ConfigBox(data)
    except BoxValueError as e:
        raise ValueError("JSON file is empty or malformed. ", e)
    except Exception as e:
        raise Exception(f"An error occurred while loading JSON file: {str(e)}")
