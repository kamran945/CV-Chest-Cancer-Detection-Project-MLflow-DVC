import logging
import os
import sys

# logging namespace
__name__ = "lungCancerDetection"

# Set up logging format and directory
log_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
log_dir = "logs"
log_file = os.path.join(log_dir, "running_logs.log")

if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

# configure
logging.basicConfig(
    level=logging.INFO,
    format=log_str,
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)
