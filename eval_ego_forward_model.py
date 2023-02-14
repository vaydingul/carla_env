import argparse

import logging

import torch
from torch.utils.data import DataLoader, Subset
from carla_env.dataset.instance import InstanceDataset
from carla_env.models.dynamic.vehicle import KinematicBicycleModel
from carla_env.evaluator.ego_forward_model import Evaluator
from utils.train_utils import get_device, seed_everything
from utils.model_utils import fetch_run_from_wandb_link, fetch_checkpoint_from_wandb_run
from utils.path_utils import create_date_time_path
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
    #                                   WANDB RUN CHECKPOINT                                #
    # ---------------------------------------------------------------------------- #
    run = fetch_run_from_wandb_link(config["wandb"]["link"])
    checkpoint = fetch_checkpoint_from_wandb_run(
        run=run, checkpoint_number=config["wandb"]["checkpoint_number"]
    )

    # ---------------------------------------------------------------------------- #
    #                                     WANDB                                    #
    # ---------------------------------------------------------------------------- #
    run_eval = create_wandb_run(config)
    # ---------------------------------------------------------------------------- #
    #                                 DATASET CLASS                                #
    # ---------------------------------------------------------------------------- #
    dataset_class = dataset_factory(run.config)

    # ---------------------------------------------------------------------------- #
    #                         TEST DATASET                                         #
    # ---------------------------------------------------------------------------- #
    dataset_test = dataset_class(config["dataset_test"])

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
                dataset_test.dilation * dataset_test.sequence_length,
            ),
        ),
        **config["dataloader_test"],
    )

    # ------------------- Log information about the dataloader -------------------- #
    logger.info(f"Test dataloader size: {len(dataloader_test)}")

    # ---------------------------------------------------------------------------- #
    #                               EGO FORWARD MODEL                              #
    # ---------------------------------------------------------------------------- #
    model_class = ego_forward_model_factory(run.config)

    # Create and initialize the model with pretrained weights and biases
    model = model_class.load_model_from_wandb_run(
        config=run.config["ego_forward_model"],
        checkpoint_path=checkpoint.name,
        device=device,
    )

    model.to(device)

    # ---------------------------------------------------------------------------- #
    #                                     LOSS                                     #
    # ---------------------------------------------------------------------------- #

    metric = metric_factory(config)

    # ------------------- Log information about the model ------------------------ #
    logger.info(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    if run_eval is not None:

        table = wandb.Table(
            columns=[
                "ID",
                "Location Metric",
                "Rotation Metric",
                "Speed Metric",
                "Image",
            ]
        )

    # ---------------------------------------------------------------------------- #
    #                                   EVALUATOR                                  #
    # ---------------------------------------------------------------------------- #
    evaluator = Evaluator(
        model=model,
        dataloader=dataloader_test,
        device=device,
        metric=metric,
        sequence_length=config["evaluation"]["sequence_length"],
        save_path=f"{config['save_path']}",
    )

    logger.info(f"Starting evaluation")

    evaluator.evaluate(run_eval, table)

    run_eval.log({"eval/Table": table})

    logger.info(f"Finished evaluation")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/volkan/Documents/Codes/carla_env/configs/ego_forward_model/evaluation/config.yml",
    )
    args = parser.parse_args()

    config = parse_yml(args.config_path)

    assert (
        config["dataset_test"]["sequence_length"]
        == config["evaluation"]["sequence_length"]
    ), "Sequence length of the dataset and the evaluation should be the same"

    main(config)
