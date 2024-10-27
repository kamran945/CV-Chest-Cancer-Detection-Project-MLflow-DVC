# CV-Chest-Cancer-Detection-Project-MLflow-DVC
Lung Cancer Detection Project (Computer Vision)
**Note on Model Training**
* This repository is primarily focused on demonstrating the integration and utilization of **Docker, Streamlit, DVC, MLflow, and Dagshub** for managing and deploying data science applications. While model training and feature engineering are essential components of machine learning workflows, this project does not emphasize those areas extensively. Instead, it serves as a practical example of how to effectively leverage these tools to streamline the machine learning lifecycle, facilitate collaboration, and create interactive applications. Therefore, users should not expect in-depth coverage of model training techniques or data feature engineering within this repo.

## Setup Steps:
* **Clone the Repository**
`git clone <repository-url>`
* **Create Conda Environment**
`conda create --name <env-name> python=3.9 -y`
* **Activate Conda Environment**
`conda activate <env-name>`

* **Install Required Libraries**
`pip install -r requirements.txt`

* **Run the following command**
`python app.y`

## Experiment Tracking and Data Management
* **Tracking Experiments on MLflow:**
    * MLflow is an open-source platform for managing the machine learning lifecycle, including experimentation, reproducibility, and deployment.
    * First Export these environment variables for dagshub mlflow tracking: `export MLFLOW_TRACKING_URI=https://mlflow.example.com` `export MLFLOW_TRACKING_USERNAME=your_username` `export MLFLOW_TRACKING_PASSWORD=your_password`
* **DVC Commands for Pipeline:**
    * DVC (Data Version Control) is a version control system for managing machine learning projects, enabling reproducible data pipelines and collaboration. 
    * DVC Commands: `dvc init` `dvc repro` `dvc dag`

## Containerization and App Deployment
* **Docker Image:**
    * Docker allows you to package applications and their dependencies into a standardized unit, ensuring consistent environments across development, testing, and production.
    * Build the Docker image: `docker build -t <image-name> .`
* **App in Streamlit:**
    * Streamlit is an open-source framework for creating web applications quickly and easily, primarily for data science projects. It allows you to turn data scripts into shareable web apps in minutes.