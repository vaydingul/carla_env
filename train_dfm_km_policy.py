from carla_env.models.dynamic.vehicle import KinematicBicycleModelV2
from carla_env.models.world.world import WorldBEVModel
from carla_env.models.policy.policy import Policy
from carla_env.models.dfm_km_policy import DecoupledForwardModelKinematicsPolicy
from carla_env.cost.masked_cost_batched import Cost

import torch
from torch.utils.data import DataLoader
import time
import logging
import wandb
import math
import argparse
from collections import deque
from utils.kinematic_utils import acceleration_to_throttle_brake
from utils.model_utils import (
    load_world_model_from_wandb_run,
    load_ego_model_from_checkpoint,
    fetch_checkpoint_from_wandb_link,
    convert_standard_bev_to_model_bev)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(config):
    # Cost function to be used
    cost = Cost(image_width=192, image_height=192, device=config.world_device)

    # Pretrained ego forward model
    ego_forward_model = load_ego_model_from_checkpoint(
        checkpoint=config.ego_forward_model_path,
        cls=KinematicBicycleModelV2,
        dt=1 / 20)
    ego_forward_model.to(device=config.ego_device)

    # Pretrained world forward model
    run = wandb.Api().run(config.wandb_link)
    checkpoint = fetch_checkpoint_from_wandb_link(
        config.wandb_link, config.checkpoint_number)

    world_forward_model, _ = load_world_model_from_wandb_run(
        run=run,
        checkpoint=checkpoint,
        cls=WorldBEVModel,
        world_model_device=config.world_device)
    world_forward_model.to(device=config.world_device)

    # Load the dataset
    # Create dataset and its loader
    data_path_train = config.data_path_train
    data_path_val = config.data_path_val
    dataset_train = InstanceDataset(
        data_path=data_path_train,
        sequence_length=run.config["num_time_step_previous"] +
        config.num_time_step_predict)
    dataset_val = InstanceDataset(
        data_path=data_path_val,
        sequence_length=run.config["num_time_step_previous"] +
        config.num_time_step_predict)

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

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup the wandb
    if config.wandb:
        run = create_resumed_run(config=config)

    else:

        run = None

    if not config.resume:

        policy_model = Policy(
            input_shape_world_state=run.config["input_shape"],
            input_shape_ego_state=4,
            action_size=2,
            hidden_size=run.config["hidden_size"])
    else:

        checkpoint = fetch_checkpoint_from_wandb_run(
            run=run)

        world_bev_model = load_world_model_from_wandb_run(
            run=run,
            checkpoint=checkpoint,
            cls=WorldBEVModel,
            world_model_device=world_model_device)

    world_bev_model.to(world_model_device)

    logger.info(
        f"Number of parameters: {sum(p.numel() for p in world_bev_model.parameters() if p.requires_grad)}")

    if not config.resume:
        world_model_optimizer = torch.optim.Adam(
            world_bev_model.parameters(), lr=config.lr)
        if config.lr_schedule:
            if isinstance(config.lr_schedule_step_size, int):
                world_model_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    world_model_optimizer,
                    step_size=config.lr_schedule_step_size,
                    gamma=config.lr_schedule_gamma)
            else:
                world_model_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    world_model_optimizer,
                    milestones=[int(s) for s in run.config["lr_schedule_step_size"].split("-")],
                    gamma=config.lr_schedule_gamma)
    else:
        world_model_optimizer = torch.optim.Adam(
            world_bev_model.parameters(), lr=run.config["lr"])
        world_model_optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"])

        if config.lr_schedule:
            if isinstance(config.lr_schedule_step_size, int):
                world_model_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    world_model_optimizer,
                    step_size=run.config["lr_schedule_step_size"],
                    gamma=run.config["lr_schedule_gamma"])
            else:
                world_model_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    world_model_optimizer,
                    milestones=[int(s) for s in run.config["lr_schedule_step_size"].split("-")],
                    gamma=run.config["lr_schedule_gamma"])

            if checkpoint["lr_scheduler_state_dict"] is not None:
                world_model_lr_scheduler.load_state_dict(
                    checkpoint["lr_scheduler_state_dict"])

    run.watch(world_bev_model)

    world_model_trainer = Trainer(
        world_bev_model,
        world_model_dataloader_train,
        world_model_dataloader_val,
        world_model_optimizer,
        world_model_device,
        num_time_step_previous=config.num_time_step_previous,
        num_time_step_future=config.num_time_step_future,
        num_epochs=config.num_epochs,
        current_epoch=checkpoint["epoch"] + 1 if config.resume else 0,
        reconstruction_loss=config.reconstruction_loss,
        logvar_clip=config.logvar_clip,
        logvar_clip_min=config.logvar_clip_min,
        logvar_clip_max=config.logvar_clip_max,
        lr_scheduler=world_model_lr_scheduler if config.lr_schedule else None,
        save_path=config.pretrained_model_path,
        train_step=checkpoint["train_step"] if config.resume else 0,
        val_step=checkpoint["val_step"] if config.resume else 0)

    logger.info("Training started!")
    world_model_trainer.learn(run)

    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Collect data from the CARLA simulator")

    parser.add_argument("--data_path_train", type=str,
                        default="data/ground_truth_bev_model_data_dummy")
    parser.add_argument("--data_path_val", type=str,
                        default="data/ground_truth_bev_model_data_dummy")

    parser.add_argument(
        "--ego_forward_model_path",
        type=str,
        default="pretrained_models/2022-09-30/17-49-06/ego_model_new.pt",
        help="Path to the forward model of the ego vehicle")

    parser.add_argument("--rollout_length", type=int, default=10)

    parser.add_argument(
        "--wandb_link",
        type=str,
        default="vaydingul/mbl/phys7134")

    parser.add_argument("--checkpoint_number", type=int, default=24)

    parser.add_argument("--ego_device", type=str, default="cuda:0",
                        help="Device to use for the forward model")

    parser.add_argument("--world_device", type=str, default="cuda:0",
                        help="Device to use for the world model")

    parser.add_argument("--mpc_device", type=str, default="cuda:0",
                        help="Device to use for the MPC module")

    config = parser.parse_args()

    main(config)
