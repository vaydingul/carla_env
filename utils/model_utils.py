import torch
import wandb


def fetch_checkpoint_from_wandb_link(wandb_link):

    # Fetch the wandb run
    api = wandb.Api()
    run = api.run(wandb_link)

    torch_files_ = [f for f in run.files() if f.name.endswith('.pt')]
    torch_files_.sort(key=lambda x: int(
        x.name.split("/")[-1].split(".")[0].split("_")[-1]))

    checkpoint = torch_files_[-1]
    checkpoint.download(replace=True)

    return checkpoint


def fetch_checkpoint_from_wandb_run(run):

    checkpoint = fetch_checkpoint_from_wandb_link(run.path)

    return checkpoint
