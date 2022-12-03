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


def load_policy_model_from_wandb_run(
        run, checkpoint, cls, policy_model_device):
    checkpoint = torch.load(
        checkpoint.name,
        map_location=policy_model_device)
    policy_model = cls(
        input_shape_world_state=run.config["input_shape_world_state"],
        input_ego_location=run.config["input_ego_location"],
        input_ego_yaw=run.config["input_ego_yaw"],
        input_ego_speed=run.config["input_ego_speed"],
        action_size=run.config["action_size"],
        hidden_size=run.config["hidden_size"],
        layers=run.config["num_layer"],
        delta_target = run.config["delta_target"])
    policy_model.load_state_dict(checkpoint["model_state_dict"])

    return policy_model, checkpoint


def load_ego_model_from_checkpoint(checkpoint, cls, dt):
    ego_forward_model = cls(dt=dt)
    ego_forward_model.load_state_dict(
        state_dict=torch.load(f=checkpoint))
    return ego_forward_model


def convert_standard_bev_to_model_bev(bev, device="cpu"):
    bev = torch.from_numpy(bev).float().to(device)
    # Permute the dimensions such that the channel dim is the first one
    bev = bev[..., [k for k in range(bev.shape[-1]) if k != 3]]
    bev = bev.permute(2, 0, 1)
    # Add offroad mask to BEV representation
    offroad_mask = torch.where(
        torch.all(
            bev == 0, dim=0), torch.ones_like(
            bev[0]), torch.zeros_like(
            bev[0]))
    bev = torch.cat([bev, offroad_mask.unsqueeze(0)], dim=0)
    bev = bev.unsqueeze(0)
    return bev
