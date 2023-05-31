from sklearn.manifold import TSNE
import argparse
from datetime import datetime
from pathlib import Path
import logging
from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader, Subset

from utilities.train_utils import get_device, seed_everything
from utilities.model_utils import fetch_run_from_wandb_link, fetch_checkpoint_from_wandb_run
from utilities.config_utils import parse_yml
from utilities.factory import *
from utilities.wandb_utils import create_wandb_run
from utilities.log_utils import get_logger, configure_logger, pretty_print_config


def plot_embeddings(embeddings, labels, name):

    import matplotlib.pyplot as plt

    # ---------------------------------------------------------------------------- #
    #                                   PLOT                                       #
    # ---------------------------------------------------------------------------- #

    # Create the figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    # Plot the embeddings
    ax.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap="jet")
    ax.set_xlabel("TSNE 1")
    ax.set_ylabel("TSNE 2")
    ax.set_title("TSNE Embeddings")
    # Add the colorbar
    cbar = fig.colorbar(ax.collections[0])

    # Save the figure
    fig.savefig(f"{name}.png")


def fit_tsne(
    model,
    dataloader_test,
    tsne,
    logger,
    device="cpu",
    num_time_step_previous=20,
    num_time_step_future=10,
    path=None,
):

    # ---------------------------------------------------------------------------- #
    #                                   TSNE FIT                                   #
    # ---------------------------------------------------------------------------- #
    logger.info("Fitting TSNE")

    # Get all the latent vectors
    latent_vectors = []
    speeds = []
    yaws = []
    bev_channel_counts = []
    for batch in tqdm(dataloader_test):
        bev = batch["bev_world"]["bev"][:, :num_time_step_previous].to(device)
        latent_vectors.append(model(bev, encoded=True))
        speeds.append(
            batch["ego"]["velocity_array"][:, :num_time_step_previous]
            .norm(2, -1, keepdim=False)
            .mean(1)
            .to(device)
        )
        yaws.append(
            batch["ego"]["rotation_array"][:, :num_time_step_previous, 2]
            .mean(1)
            .to(device)
        )

        bev_channel_counts.append(bev.sum([3, 4]).mean(1).to(device))

    latent_vectors = torch.cat(latent_vectors, dim=0)
    speeds = torch.cat(speeds, dim=0).cpu().detach().numpy()
    yaws = torch.cat(yaws, dim=0).cpu().detach().numpy()
    bev_channel_counts = torch.cat(bev_channel_counts, dim=0).cpu().detach().numpy()

    # Flatten the latent vectors
    latent_vectors = latent_vectors.view(latent_vectors.shape[0], -1)

    logger.info(f"Latent vectors shape: {latent_vectors.shape}")

    # Fit the tsne
    embedded_latent_vectors = tsne.fit_transform(latent_vectors.cpu().detach().numpy())

    # Plot the embeddings
    fig = plot_embeddings(embedded_latent_vectors, speeds, os.path.join(path, "speed"))
    fig = plot_embeddings(embedded_latent_vectors, yaws, os.path.join(path, "yaw"))
    for k in range(bev.shape[2]):
        fig = plot_embeddings(
            embedded_latent_vectors,
            bev_channel_counts[:, k],
            os.path.join(path, f"bev_channel_{k}"),
        )


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
    #                               WORLD FORWARD MODEL                            #
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

    path = os.path.join(
        config["save_path"], "-".join(f"{k}_{v}" for k, v in config["tsne"].items())
    )
    os.makedirs(path, exist_ok=True)

    fit_tsne(
        model,
        dataloader_test,
        tsne,
        logger,
        device=device,
        num_time_step_previous=config["num_time_step_previous"],
        num_time_step_future=config["num_time_step_future"],
        path=path,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/volkan/Documents/Codes/carla_env/configs/tsne/config_extended_bev.yml",
    )
    args = parser.parse_args()

    config = parse_yml(args.config_path)

    config["config_path"] = args.config_path

    main(config)
