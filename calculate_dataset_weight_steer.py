from carla_env.dataset.instance import InstanceDataset
import os
import logging
import torch
import argparse
from tqdm import tqdm
from utilities.config_utils import parse_yml
from utilities.factory import *
from utilities.log_utils import configure_logger, get_logger


def main(config):
    # ---------------------------------------------------------------------------- #
    #                                    LOGGER                                    #
    # ---------------------------------------------------------------------------- #
    logger = get_logger(__name__)
    configure_logger(__name__, log_path=config["log_path"], log_level=logging.INFO)

    # ---------------------------------------------------------------------------- #
    #                                 DATASET CLASS                                #
    # ---------------------------------------------------------------------------- #
    dataset_class = dataset_factory(config)
    dataset = dataset_class(config["dataset"]["config"])
    logger.info(f"Dataset size: {len(dataset)}")

    # -------------------------------- Weight Path ------------------------------- #
    weight_path = f"{config['dataset']['config']['data_path']}/{config['num_time_step_previous']}-{config['num_time_step_future']}-{config['dilation']}-{'-'.join(config['read_keys'])}-STEER-weights.pt"

    if os.path.exists(weight_path):
        logger.info("Loading weights from file")
        weights = torch.load(weight_path)
    else:
        logger.info("Calculating weights")
        weights = torch.Tensor(
            [
                dataset.__get_weight_steer__(k, config["weight_coefficients"])
                for k in tqdm(range(len(dataset)))
            ]
        )
        torch.save(weights, weight_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default="configs/dataset_weight/config_steer.yml"
    )

    config = parser.parse_args()
    args = parser.parse_args()

    config = parse_yml(args.config_path)
    config["config_path"] = args.config_path

    main(config)
