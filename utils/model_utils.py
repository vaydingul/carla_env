import wandb


def fetch_checkpoint_from_wandb_link(wandb_link, epoch=None):

    # Fetch the wandb run
    api = wandb.Api()
    run = api.run(wandb_link)

    torch_files_ = [f for f in run.files() if f.name.endswith('.pt')]
    torch_files_.sort(key=lambda x: abs(int(
        x.name.split("/")[-1].split(".")[0].split("_")[-1]) - epoch))

    checkpoint = torch_files_[0]
    checkpoint.download(replace=True)

    return checkpoint


def fetch_checkpoint_from_wandb_run(run, epoch=None):

    checkpoint = fetch_checkpoint_from_wandb_link(run.path, epoch)

    return checkpoint
