from sklearn.manifold import TSNE
import argparse
from datetime import datetime
from pathlib import Path
import logging
import wandb

import torch
from torch.utils.data import DataLoader, Subset

from utils.train_utils import get_device, seed_everything
from utils.model_utils import fetch_run_from_wandb_link, fetch_checkpoint_from_wandb_run
from utils.path_utils import create_date_time_path
from utils.config_utils import parse_yml
from utils.factory import *
from utils.wandb_utils import create_wandb_run
from utils.log_utils import get_logger, configure_logger, pretty_print_config
import wandb


def fit_tsne(
    model,
    dataloader_test,
    tsne,
    logger,
    device="cpu",
    num_time_step_previous=20,
    num_time_step_future=10,
):

    # ---------------------------------------------------------------------------- #
    #                                   TSNE FIT                                   #
    # ---------------------------------------------------------------------------- #
    logger.info("Fitting TSNE")

    # Get all the latent vectors
    latent_vectors = []
    for batch in dataloader_test:
        latent_vectors.append(
            model(batch["bev_world"]["bev"][:, :num_time_step_previous].to(device))
        )
    latent_vectors = torch.cat(latent_vectors, dim=0)

    # Flatten the latent vectors
    latent_vectors = latent_vectors.view(latent_vectors.shape[0], -1)

    # Fit the tsne
    embedded_latent_vectors = tsne.fit_transform(latent_vectors.cpu().detach().numpy())

    return tsne


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
    device = "cpu"  # get_device()

    # ---------------------------------------------------------------------------- #
    #                                   WANDB RUN CHECKPOINT                                #
    # ---------------------------------------------------------------------------- #
    run = fetch_run_from_wandb_link(config["wandb_world_forward_model"]["link"])
    checkpoint_object = fetch_checkpoint_from_wandb_run(
        run=run,
        checkpoint_number=config["wandb_world_forward_model"]["checkpoint_number"],
    )
    checkpoint_path = checkpoint_object.name
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # ---------------------------------------------------------------------------- #
    #                                 DATASET CLASS                                #
    # ---------------------------------------------------------------------------- #
    dataset_class = dataset_factory(run.config)

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
                config["analysis"]["test_step"],
            ),
        ),
        **config["dataloader_test"],
    )

    # ------------------- Log information about the dataloader -------------------- #
    logger.info(f"Test dataloader size: {len(dataloader_test)}")

    # ---------------------------------------------------------------------------- #
    #                               WORLD FORWARD MODEL                              #
    # ---------------------------------------------------------------------------- #
    model_class = world_forward_model_factory(run.config)

    # Create and initialize the model with pretrained weights and biases
    model = model_class.load_model_from_wandb_run(
        config=run.config["world_forward_model"]["config"],
        checkpoint_path=checkpoint_path,
        device=device,
    )

    model.to(device)

    # ------------------- Log information about the model ------------------------ #
    logger.info(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # ---------------------------------------------------------------------------- #
    #                                  TSNE Object                                 #
    # ---------------------------------------------------------------------------- #
    tsne = TSNE(
        **config["tsne"],
    )

    fit_tsne(
        model,
        dataloader_test,
        tsne,
        logger,
        device=device,
        num_time_step_previous=config["num_time_step_previous"],
        num_time_step_future=config["num_time_step_future"],
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/vaydingul/Documents/Codes/carla_env/configs/tsne/config.yml",
    )
    args = parser.parse_args()

    config = parse_yml(args.config_path)

    config["config_path"] = args.config_path

    main(config)
