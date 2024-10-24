import setuptools

# Get the long description from the README file
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

REPO_NAME = "CV-Chest-Cancer-Detection-Project-MLflow-DVC"
SRC_REPO_NAME = "lungCancerDetection"

AUTHOR_NAME = "kamran"
AUTHOR_EMAIL = "kamran945@gmail.com"

__version__ = "0.0.0"

setuptools.setup(
    name=SRC_REPO_NAME,
    version=__version__,
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    description="Lung Cancer Detection.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
