import wandb


def fetch_checkpoint_from_wandb_link(wandb_link, checkpoint_number=-1):

    # Fetch the wandb run
    api = wandb.Api()
    run = api.run(wandb_link)

    torch_files_ = [f for f in run.files() if f.name.endswith('.pt')]
    torch_files_.sort(key=lambda x: abs(int(x.name.split(
        "/")[-1].split(".")[0].split("_")[-1]) - checkpoint_number if checkpoint_number >= 0 else 0))

    checkpoint = torch_files_[0 if checkpoint_number >= 0 else -1]
    checkpoint.download(replace=True)

    return checkpoint


def fetch_checkpoint_from_wandb_run(run, checkpoint_number=-1):

    checkpoint = fetch_checkpoint_from_wandb_link(run.path, checkpoint_number)

    return checkpoint
