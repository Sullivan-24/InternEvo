#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import logging
import os

LOGGER_NAME = "internlm"
LOGGER_FORMAT = "%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s in %(funcName)s -- %(message)s"
LOGGER_LEVEL = "info"
LOGGER_LEVEL_CHOICES = ["debug", "info", "warning", "error", "critical"]
LOGGER_LEVEL_HELP = (
    "The logging level threshold, choices=['debug', 'info', 'warning', 'error', 'critical'], default='info'"
)

std_logger = None


def get_logger(
    logger_name: str = LOGGER_NAME,
    logging_level: str = LOGGER_LEVEL,
    launch_time: str = None,
    job_name: str = None,
    file_name: str = None,
) -> logging.Logger:
    """Configure the logger that is used for uniscale framework.

    Args:
        logger_name (str): used to create or get the correspoding logger in
            getLogger call. It will be "internlm" by default.
        logging_level (str, optional): Logging level in string or logging enum.

    Returns:
        logger (logging.Logger): the created or modified logger.

    """
    global std_logger
    if std_logger is not None:
        return std_logger

    logger = logging.getLogger(logger_name)

    if logging_level not in LOGGER_LEVEL_CHOICES:
        logging_level = LOGGER_LEVEL
        print(LOGGER_LEVEL_HELP)

    logging_level = logging.getLevelName(logging_level.upper())

    # add stream handler
    handler = logging.StreamHandler()
    handler.setLevel(logging_level)
    logger.setLevel(logging_level)
    handler.setFormatter(logging.Formatter(LOGGER_FORMAT))
    logger.addHandler(handler)

    # add file handler
    if file_name is not None:
        log_folder = os.path.join("RUN", job_name, launch_time, "logs")
        log_filepath = os.path.join(log_folder, file_name)
        try:
            os.makedirs(log_folder, exist_ok=True)
            # Set full permissions on the created directory
            try:
                os.chmod(log_folder, 0o777)
            except:
                pass
        except (FileExistsError, PermissionError, OSError):
            # Ignore if directory already exists or permission denied
            # Multiple ranks may try to create concurrently
            pass

        try:
            # Small delay to ensure directory is visible across all processes
            import time
            time.sleep(0.5)
            filehandler = logging.FileHandler(log_filepath)
            filehandler.setLevel(logging_level)
            filehandler.setFormatter(logging.Formatter(LOGGER_FORMAT))
            logger.addHandler(filehandler)
        except (FileNotFoundError, PermissionError, OSError):
            # If file creation fails, just use console logging
            pass

        std_logger = logger

    return logger
