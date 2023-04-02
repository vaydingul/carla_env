import argparse
from datetime import datetime
from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader, Subset

from utils.train_utils import get_device, seed_everything
from utils.model_utils import fetch_run_from_wandb_link, fetch_checkpoint_from_wandb_run
from utils.config_utils import parse_yml
from utils.factory import *
from utils.wandb_utils import create_wandb_run
from utils.log_utils import get_logger, configure_logger, pretty_print_config
import wandb


def main(config):

    # ---------------------------------------------------------------------------- #
    #                                    LOGGER                                    #
    # ---------------------------------------------------------------------------- #
    logger = get_logger(__name__)
    configure_logger(__name__, log_path=config["log_path"], log_level=logging.INFO)
    pretty_print_config(logger, config)
    # ---------------------------------------------------------------------------- #
    #                                     SEED                                     #
    # ---------------------------------------------------------------------------- #
    seed_everything(config["seed"])

    # ---------------------------------------------------------------------------- #
    #                                    DEVICE                                    #
    # ---------------------------------------------------------------------------- #
    device = get_device()

    # ---------------------------------------------------------------------------- #
    #                                     WANDB                                    #
    # ---------------------------------------------------------------------------- #
    run_eval = create_wandb_run(config)

    # ---------------------------------------------------------------------------- #
    #                                 DATASET CLASS                                #
    # ---------------------------------------------------------------------------- #
    dataset_class = dataset_factory(run_eval.config)

    # ---------------------------------------------------------------------------- #
    #                         TEST DATASET                                         #
    # ---------------------------------------------------------------------------- #
    dataset_test = dataset_class(config["dataset_test"]["config"])

    # --------------------- Log information about the dataset -------------------- #
    logger.info(f"Test dataset size: {len(dataset_test)}")

    # ---------------------------------------------------------------------------- #
    #                       TEST DATALOADER                                        #
    # ---------------------------------------------------------------------------- #
    dataloader_test = DataLoader(
        Subset(
            dataset_test,
            range(
                0,
                len(dataset_test),
                config["evaluation"]["test_step"],
            ),
        ),
        **config["dataloader_test"],
    )

    # ------------------- Log information about the dataloader -------------------- #
    logger.info(f"Test dataloader size: {len(dataloader_test)}")


    
    run_eval.save(config["config_path"])

    # ---------------------------------------------------------------------------- #
    #                                   EVALUATOR                                  #
    # ---------------------------------------------------------------------------- #
    evaluator_class = evaluator_factory(config)

    evaluator = evaluator_class(
        dataset=dataset_test,
        dataloader=dataloader_test,
        device=device,
        renderer=config["evaluation"]["renderer"],
        metrics=config["evaluation"]["metrics"],
        num_time_step_previous=config["evaluation"]["num_time_step_previous"],
        num_time_step_predict=config["evaluation"]["num_time_step_predict"],
        thresholds=config["evaluation"]["thresholds"],
        wandb_log_interval=config["evaluation"]["wandb_log_interval"],
        save_path=f"{config['save_path']}",
    )

    logger.info(f"Starting evaluation")

    evaluator.evaluate(run_eval)

    logger.info(f"Finished evaluation")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/volkan/Documents/Codes/carla_env/configs/world_forward_model/evaluation/config_last_frame_repeated.yml",
    )
    args = parser.parse_args()

    config = parse_yml(args.config_path)
    # config["save_path"] = create_date_time_path(config["save_path"])
    config["config_path"] = args.config_path

    assert (
        config["dataset_test"]["config"]["sequence_length"]
        == config["evaluation"]["sequence_length"]
    ), "Sequence length of the dataset and the evaluator should be the same"

    main(config)