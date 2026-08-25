import logging
import os
from datetime import datetime


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(
    LOG_DIR,
    f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
)


logging.basicConfig(
    filename=LOG_FILE,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


console_handler = logging.StreamHandler()

console_handler.setFormatter(
    logging.Formatter(
        "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
    )
)

logging.getLogger().addHandler(console_handler)