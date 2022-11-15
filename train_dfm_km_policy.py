from carla_env.models.dynamic.vehicle import KinematicBicycleModelV2
from carla_env.models.world.world import WorldBEVModel
from carla_env.models.policy.policy import Policy
from carla_env.models.dfm_km_policy import DecoupledForwardModelKinematicsPolicy
from carla_env.cost.masked_cost_batched import Cost
from carla_env.trainer.dfm_km_policy import Trainer
import torch
from torch.utils.data import DataLoader
import time
import logging
import wandb
import math
import argparse
from collections import deque

from utils.model_utils import (
    load_world_model_from_wandb_run,
    load_policy_model_from_wandb_run,
    load_ego_model_from_checkpoint,
    fetch_checkpoint_from_wandb_link)

from utils.train_utils import seed_everything

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(config):

    seed_everything(seed=config.seed)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    # Cost function to be used
    cost = Cost(image_width=192, image_height=192, device=config.world_device)

    # Pretrained ego forward model
    ego_forward_model = load_ego_model_from_checkpoint(
        checkpoint=config.ego_forward_model_path,
        cls=KinematicBicycleModelV2,
        dt=1 / 20)
    ego_forward_model.to(device=device)

    # Pretrained world forward model
    world_model_run = wandb.Api().world_model_run(
        config.world_forward_model_wandb_link)
    checkpoint = fetch_checkpoint_from_wandb_link(
        config.world_forward_model_wandb_link, config.checkpoint_number)

    world_forward_model, _ = load_world_model_from_wandb_run(
        run=world_model_run,
        checkpoint=checkpoint,
        cls=WorldBEVModel,
        world_model_device=device)
    world_forward_model.to(device=device)

    # Load the dataset
    # Create dataset and its loader
    data_path_train = config.data_path_train
    data_path_val = config.data_path_val
    dataset_train = InstanceDataset(
        data_path=data_path_train,
        sequence_length=world_model_run.config["num_time_step_previous"] +
        config.num_time_step_predict,
        read_keys=["bev", "ego"])
    dataset_val = InstanceDataset(
        data_path=data_path_val,
        sequence_length=world_model_run.config["num_time_step_previous"] +
        config.num_time_step_predict,
        read_keys=["bev", "ego"])

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True)
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=True)

    logger.info(f"Train dataset size: {len(dataset_train)}")
    logger.info(f"Val dataset size: {len(dataset_val)}")
    # Setup the wandb
    if config.wandb:

        if not config.resume:

            run = create_initial_run(config=config)

        else:

            run = create_resumed_run(config=config)

    else:

        run = None

    if not config.resume:

        policy_model = Policy(
            input_shape_world_state=world_model_run.config["input_shape"],
            input_shape_ego_state=config.input_shape_ego_state,
            action_size=config.action_size,
            hidden_size=world_model_run.config["hidden_size"])
    else:

        checkpoint = fetch_checkpoint_from_wandb_run(
            run=run)

        policy_model = load_policy_model_from_wandb_run(
            run=run,
            checkpoint=checkpoint,
            cls=Policy,
            policy_model_device=device)

    model = DecoupledForwardModelKinematicsPolicy(
        world_forward_model=world_forward_model,
        ego_forward_model=ego_forward_model,
        policy_model=policy_model,
        device=device)

    if not config.resume:
        optimizer = torch.optim.Adam(
            policy_model.parameters(), lr=config.lr)
        if config.lr_schedule:
            if isinstance(config.lr_schedule_step_size, int):
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=config.lr_schedule_step_size,
                    gamma=config.lr_schedule_gamma)
            else:
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[
                        int(s) for s in run.config["lr_schedule_step_size"].split("-")],
                    gamma=config.lr_schedule_gamma)
    else:
        optimizer = torch.optim.Adam(
            policy_model.parameters(), lr=run.config["lr"])
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"])

        if config.lr_schedule:
            if isinstance(config.lr_schedule_step_size, int):
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=run.config["lr_schedule_step_size"],
                    gamma=run.config["lr_schedule_gamma"])
            else:
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[
                        int(s) for s in run.config["lr_schedule_step_size"].split("-")],
                    gamma=run.config["lr_schedule_gamma"])

            if checkpoint["lr_scheduler_state_dict"] is not None:
                lr_scheduler.load_state_dict(
                    checkpoint["lr_scheduler_state_dict"])

    run.watch(policy_model)

    trainer = Trainer(
        model,
        dataloader_train,
        dataloader_val,
        optimizer,
        device,
        cost,
        num_time_step_previous=world_model_run.config["num_time_step_previous"],
        num_time_step_future=world_model_run.config["num_time_step_future"],
        num_epochs=config.num_epochs,
        current_epoch=checkpoint["epoch"] +
        1 if config.resume else 0,
        lr_scheduler=lr_scheduler if config.lr_schedule else None,
        save_path=config.pretrained_model_path,
        train_step=checkpoint["train_step"] if config.resume else 0,
        val_step=checkpoint["val_step"] if config.resume else 0)

    logger.info("Training started!")
    trainer.learn(run)

    run.finish()


if __name__ == "__main__":

    checkpoint_path = Path("pretrained_models")

    date_ = Path(datetime.today().strftime('%Y-%m-%d'))
    time_ = Path(datetime.today().strftime('%H-%M-%S'))

    checkpoint_path = checkpoint_path / date_ / time_
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Collect data from the CARLA simulator")

    # TRAINING PARAMETERS
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_path_train", type=str,
                        default="data/ground_truth_bev_model_data_dummy")
    parser.add_argument("--data_path_val", type=str,
                        default="data/ground_truth_bev_model_data_dummy")
    parser.add_argument("--pretrained_model_path",
                        type=str, default=checkpoint_path)
    parser.add_argument(
        "--resume",
        type=lambda x: (
            str(x).lower() == 'true'),
        default=False)

    parser.add_argument("--reconstruction_loss", type=str, default="mse_loss")
    parser.add_argument("--logvar_clip", type=lambda x: (
        str(x).lower() == 'true'), default=True)
    parser.add_argument("--logvar_clip_min", type=float, default=-5)
    parser.add_argument("--logvar_clip_max", type=float, default=5)
    parser.add_argument("--lr_schedule", type=lambda x: (
        str(x).lower() == 'true'), default=True)
    parser.add_argument("--lr_schedule_step_size", default=5)
    parser.add_argument("--lr_schedule_gamma", type=float, default=0.5)

    # POLICY MODEL PARAMETERS
    parser.add_argument("--input_shape_ego_state", type=int, default=4)
    parser.add_argument("--action_size", type=int, default=2)

    # WANDB RELATED PARAMETERS
    parser.add_argument(
        "--wandb",
        type=lambda x: (
            str(x).lower() == 'true'),
        default=True)
    parser.add_argument("--wandb_project", type=str, default="mbl")
    parser.add_argument(
        "--wandb_group",
        type=str,
        default="world-forward-model-multi-step")
    parser.add_argument("--wandb_name", type=str, default="model")
    parser.add_argument("--wandb_id", type=str, default=None)

    parser.add_argument(
        "--ego_forward_model_path",
        type=str,
        default="pretrained_models/2022-09-30/17-49-06/ego_model_new.pt",
        help="Path to the forward model of the ego vehicle")

    parser.add_argument(
        "--world_forward_model_wandb_link",
        type=str,
        default="vaydingul/mbl/phys7134")

    parser.add_argument(
        "--world_forward_model_checkpoint_number",
        type=int,
        default=24)

    config = parser.parse_args()

    main(config)
