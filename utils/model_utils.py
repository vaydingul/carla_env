import wandb
import os
import torch
import logging

logger = logging.getLogger(__name__)


def fetch_checkpoint_from_wandb_link(wandb_link, checkpoint_number=-1):

    # Fetch the wandb run
    api = wandb.Api()
    run = api.run(wandb_link)

    torch_files_ = [f for f in run.files() if f.name.endswith(".pt")]
    torch_files_.sort(
        key=lambda x: abs(
            int(x.name.split("/")[-1].split(".")[0].split("_")[-1])
            - (checkpoint_number if checkpoint_number >= 0 else 0)
        )
    )

    checkpoint = torch_files_[0 if checkpoint_number >= 0 else -1]

    logger.info(f"Fetching checkpoint {checkpoint.name} from wandb run {wandb_link}")

    # Check if it exists, if not, download it
    if not os.path.exists(checkpoint.name):
        checkpoint.download(replace=True)

    return checkpoint


def fetch_checkpoint_from_wandb_run(run, checkpoint_number=-1):

    if isinstance(run.path, list):
        wandb_link = "/".join(run.path)
    else:
        wandb_link = run.path

    checkpoint = fetch_checkpoint_from_wandb_link(wandb_link, checkpoint_number)

    return checkpoint


def fetch_run_from_wandb_link(wandb_link):
    api = wandb.Api()
    run = api.run(wandb_link)
    return run


def convert_standard_bev_to_model_bev(
    bev,
    agent_channel=7,
    vehicle_channel=6,
    selected_channels=[0, 1, 2, 3, 4, 5, 6, 11],
    calculate_offroad=False,
    device="cpu",
):

    bev_ = torch.from_numpy(bev).float()
    # Permute the dimensions such that the channel dim is the first one
    agent_mask = bev_[..., agent_channel]
    bev_[..., vehicle_channel] = torch.logical_and(
        bev_[..., vehicle_channel], torch.logical_not(bev_[..., agent_channel])
    )

    bev = bev_[..., selected_channels]

    bev = bev.permute(2, 0, 1)
    # Add offroad mask to BEV representation
    if calculate_offroad:
        offroad_mask = torch.where(
            torch.all(bev == 0, dim=0),
            torch.ones_like(bev[0]),
            torch.zeros_like(bev[0]),
        )
        bev = torch.cat([bev, offroad_mask.unsqueeze(0)], dim=0)
    bev = bev.unsqueeze(0).to(device)
    return bev
