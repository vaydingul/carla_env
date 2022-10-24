import torch
import wandb


def fetch_model_from_wandb(wandb_link):

    # Fetch the wandb run
    api = wandb.Api()
    run = api.run(wandb_link)

    torch_files_ = [f for f in run.files() if f.name.endswith('.pt')]
    torch_files_.sort(key=lambda x: int(
        x.name.split("/")[-1].split(".")[0].split("_")[-1]))

    model_file = torch_files_[-1]
    model_file.download(replace=True)

    return model_file, run
