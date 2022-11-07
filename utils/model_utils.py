import wandb
import os
import torch


def fetch_checkpoint_from_wandb_link(wandb_link, checkpoint_number=-1):

    # Fetch the wandb run
    api = wandb.Api()
    run = api.run(wandb_link)

    torch_files_ = [f for f in run.files() if f.name.endswith('.pt')]
    torch_files_.sort(key=lambda x: abs(int(x.name.split(
        "/")[-1].split(".")[0].split("_")[-1]) - checkpoint_number if checkpoint_number >= 0 else 0))

    checkpoint = torch_files_[0 if checkpoint_number >= 0 else -1]
    # Check if it exists, if not, download it
    if not os.path.exists(checkpoint.name):
        checkpoint.download(replace=True)

    return checkpoint


def fetch_checkpoint_from_wandb_run(run, checkpoint_number=-1):

    checkpoint = fetch_checkpoint_from_wandb_link(run.path, checkpoint_number)

    return checkpoint


def load_world_model_from_wandb_run(
        run,
        checkpoint,
        cls,
        world_model_device):

    checkpoint = torch.load(
        checkpoint.name,
        map_location=world_model_device)
    world_bev_model = cls(
        input_shape=run.config["input_shape"],
        hidden_channel=run.config["hidden_channel"],
        output_channel=run.config["output_channel"],
        num_encoder_layer=run.config["num_encoder_layer"],
        num_probabilistic_encoder_layer=run.config[
            "num_probabilistic_encoder_layer"],
        num_time_step=run.config["num_time_step_previous"] + 1,
        dropout=run.config["dropout"])
    world_bev_model.load_state_dict(checkpoint["model_state_dict"])

    return world_bev_model, checkpoint


def load_ego_model_from_checkpoint(checkpoint, cls, dt):
    ego_forward_model = cls(dt=dt)
    ego_forward_model.load_state_dict(
        state_dict=torch.load(f=checkpoint))
