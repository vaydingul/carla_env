import logging
from utils.path_utils import create_date_time_path


def get_logger(name=__name__):
    logger = logging.getLogger(name)
    return logger


def configure_logger(name, log_path=None, log_level=logging.INFO):

    if log_path is not None:
        log_path = create_date_time_path(log_path)
        log_path = log_path / f"{name}.log"

    logging.basicConfig(
        filename=log_path,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d ==> %(message)s",
    )
