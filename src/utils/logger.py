import logging
import os  # CHANGED(baseline): added for the project-root log dir below
from logging import FileHandler, StreamHandler

# CHANGED(baseline): compute log directory relative to the project root (one level up from src/),
# so the logger works regardless of cwd (was the cwd-relative './log/').
_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(_LOG_DIR, exist_ok=True)


def get_logger(name: str):
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    file_handler = FileHandler(os.path.join(_LOG_DIR, f'{name}.log'))  # CHANGED(baseline): was './log/{name}.log'
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
