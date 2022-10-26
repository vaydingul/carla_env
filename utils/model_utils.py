import torch
import wandb


def fetch_model_from_wandb_link(wandb_link):

    # Fetch the wandb run
    api = wandb.Api()
    run = api.run(wandb_link)

    model_file, run = fetch_model_from_wandb_run(run)

    return model_file, run


def fetch_model_from_wandb_run(run):

    torch_files_ = [f for f in run.files() if f.name.endswith('.pt')]
    torch_files_.sort(key=lambda x: int(
        x.name.split("/")[-1].split(".")[0].split("_")[-1]))

    model_file = torch_files_[-1]
    model_file.download(replace=True)

    return model_file, run
