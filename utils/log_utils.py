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


def get_max_key_length(d):
    # Get the max length of the keys
    max_key_len = max([len(key) for key in d.keys()])
    return max_key_len


def pretty_print_config(logger, config):
    """Print the whole config dictionary (it might be nested) in a table format"""

    # Get the max length of the keys
    max_key_len = get_max_key_length(config)

    # Print the config
    logger.info("Config:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"{key.ljust(max_key_len)}:")
            max_key_len_ = get_max_key_length(value)
            for key_, value_ in value.items():
                logger.info(f"    {key_.ljust(max_key_len_)}: {value_}")
        else:
            logger.info(f"{key.ljust(max_key_len)}: {value}")
