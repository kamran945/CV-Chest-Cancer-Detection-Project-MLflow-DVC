import os
from pathlib import Path
import logging


PROJECT_REPO = "lungCancerDetection"

file_list = [
    ".github/workflows/.gitkeep",
    f"src/{PROJECT_REPO}/__init__.py",
    f"src/{PROJECT_REPO}/config/__init__.py",
    f"src/{PROJECT_REPO}/config/configuration.py",
    f"src/{PROJECT_REPO}/components/__init__.py",
    f"src/{PROJECT_REPO}/entity/__init__.py",
    f"src/{PROJECT_REPO}/utils/__init__.py",
    f"src/{PROJECT_REPO}/utils/common.py",
    f"src/{PROJECT_REPO}/constants/__init__.py",
    f"src/{PROJECT_REPO}/pipeline/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "dvc.yaml",
    "README.md",
    "main.py",
    "app.py",
    "Dockerfile",
    "setup.py",
    "research/trials.ipynb",
    "requirements.txt",
    "templates/index.html",
]

for filepath in file_list:
    filepath = Path(filepath)
    directory, filenames = os.path.split(filepath)

    if directory != "":
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory {directory}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as file:
            pass
            logging.info(f"Created file {filepath}")

    else:
        logging.info(f"File {filepath} already exists")
