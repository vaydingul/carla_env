import logging
from pprint import pprint
from carla_env.models.dynamic.vehicle import KinematicBicycleModel
from carla_env.models.world.world import WorldBEVModel
from carla_env.models.policy.policy_fused import Policy
from carla_env.models.dfm_km_cp import DecoupledForwardModelKinematicsCoupledPolicy
from carla_env.cost.masked_cost_batched_policy_extended_bev import Cost
from carla_env.trainer.dfm_km_cp_extended_bev_ddp_gt_bev_encoded_policy_fused import (
    Trainer,
)
from carla_env.dataset.instance import InstanceDataset
import torch
from torch.utils.data import DataLoader
import time
import wandb
import math
import argparse
from datetime import datetime
from pathlib import Path
from collections import deque

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import os

from utilities.model_utils import (
    load_world_model_from_wandb_run,
    load_policy_model_from_wandb_run,
    load_ego_model_from_checkpoint,
    fetch_checkpoint_from_wandb_link,
    fetch_checkpoint_from_wandb_run,
)

from utilities.train_utils import seed_everything, get_device

from utilities.wandb_utils import create_initial_run, create_resumed_run

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d ==> %(message)s",
)


def ddp_setup(rank, world_size, master_port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def create_wandb_run(config):
    # Setup the wandb
    if config.wandb:

        if not config.resume:

            run = create_initial_run(config=config)

        else:

            run = create_resumed_run(config=config)

    else:

        run = None

    return run


def main(rank, world_size, run, config):

    ddp_setup(rank, world_size, config.master_port)

    seed_everything(seed=config.seed)

    # ---------------------------------------------------------------------------- #
    #                                 Cost Function                                #
    # ---------------------------------------------------------------------------- #
    cost = Cost(image_width=192, image_height=192, device=rank)

    # ---------------------------------------------------------------------------- #
    #                         Pretrained ego forward model                         #
    # ---------------------------------------------------------------------------- #
    ego_forward_model = load_ego_model_from_checkpoint(
        checkpoint=config.ego_forward_model_path, cls=KinematicBicycleModel, dt=1 / 20
    )
    ego_forward_model.to(device=rank)

    # ---------------------------------------------------------------------------- #
    #                        Pretrained world forward model                        #
    # ---------------------------------------------------------------------------- #
    world_model_run = wandb.Api().run(config.world_forward_model_wandb_link)
    checkpoint = fetch_checkpoint_from_wandb_link(
        config.world_forward_model_wandb_link,
        config.world_forward_model_checkpoint_number,
    )
    world_forward_model = WorldBEVModel.load_model_from_wandb_run(
        run=world_model_run,
        checkpoint=checkpoint,
        device={f"cuda:0": f"cuda:{rank}"} if config.num_gpu > 1 else rank,
    )
    world_forward_model.to(device=rank)

    config.num_time_step_previous = (
        world_model_run.config["num_time_step_previous"]
        if config.num_time_step_previous < 0
        else config.num_time_step_previous
    )
    config.num_time_step_future = (
        world_model_run.config["num_time_step_future"]
        if config.num_time_step_future < 0
        else config.num_time_step_future
    )

    # ---------------------------------------------------------------------------- #
    #                                    Dataset                                   #
    # ---------------------------------------------------------------------------- #
    # Load the dataset
    # Create dataset and its loader
    data_path_train = config.data_path_train
    data_path_val = config.data_path_val
    dataset_train = InstanceDataset(
        data_path=data_path_train,
        sequence_length=config.num_time_step_previous + config.num_time_step_future,
        read_keys=["bev_world", "ego", "navigation_downsampled", "occ"],
        dilation=config.dataset_dilation,
        bev_agent_channel=7,
        bev_vehicle_channel=6,
        bev_selected_channels=[0, 1, 2, 3, 4, 5, 6, 11],
        bev_calculate_offroad=False,
    )
    dataset_val = InstanceDataset(
        data_path=data_path_val,
        sequence_length=config.num_time_step_previous + config.num_time_step_future,
        read_keys=["bev_world", "ego", "navigation_downsampled", "occ"],
        dilation=config.dataset_dilation,
        bev_agent_channel=7,
        bev_vehicle_channel=6,
        bev_selected_channels=[0, 1, 2, 3, 4, 5, 6, 11],
        bev_calculate_offroad=False,
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=True,
        sampler=DistributedSampler(dataset_train, shuffle=True),
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=True,
        sampler=DistributedSampler(dataset_val, shuffle=False),
    )

    logger.info(f"Train dataset size: {len(dataset_train)}")
    logger.info(f"Val dataset size: {len(dataset_val)}")

    # ---------------------------------------------------------------------------- #
    #                                  Policy Model                                 #
    # ---------------------------------------------------------------------------- #
    if not config.resume:

        c, h, w = world_forward_model.world_previous_bev_encoder.to(
            "cpu"
        ).get_output_shape()

        _input_shape_world_state = (c, h, w)

        if config.wandb:
            run.config.update({"input_shape_world_state": _input_shape_world_state})

        policy_model = Policy(
            input_shape_world_state=_input_shape_world_state,
            input_ego_location=config.input_ego_location,
            input_ego_yaw=config.input_ego_yaw,
            input_ego_speed=config.input_ego_speed,
            action_size=config.action_size,
            hidden_size=config.hidden_size,
            occupancy_size=config.occupancy_size,
            layers=config.num_layer,
            delta_target=config.delta_target,
            dropout=config.dropout,
        )

    else:

        checkpoint = fetch_checkpoint_from_wandb_run(
            run=run, checkpoint_number=config.resume_checkpoint_number
        )

        policy_model = Policy.load_model_from_wandb_run(
            run=run,
            checkpoint=checkpoint,
            device={f"cuda:0": f"cuda:{rank}"} if config.num_gpu > 1 else rank,
        )
    policy_model.to(device=rank)
    # ---------------------------------------------------------------------------- #
    #                              DFM_KM with Policy                              #
    # ---------------------------------------------------------------------------- #
    model = DecoupledForwardModelKinematicsCoupledPolicy(
        ego_model=ego_forward_model,
        world_model=world_forward_model,
        policy_model=policy_model,
    )
    model.to(device=rank)
    # ---------------------------------------------------------------------------- #

    # ---------------------------------------------------------------------------- #
    #                            Optimizer and Scheduler                           #
    # ---------------------------------------------------------------------------- #
    if not config.resume:
        optimizer = torch.optim.Adam(policy_model.parameters(), lr=config.lr)
        if config.lr_schedule:
            if isinstance(config.lr_schedule_step_size, int):
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=config.lr_schedule_step_size,
                    gamma=config.lr_schedule_gamma,
                )
            else:
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[
                        int(s) for s in run.config["lr_schedule_step_size"].split("-")
                    ],
                    gamma=config.lr_schedule_gamma,
                )
    else:

        checkpoint = torch.load(
            checkpoint.name,
            map_location=f"cuda:{rank}" if isinstance(rank, int) else rank,
        )

        optimizer = torch.optim.Adam(policy_model.parameters(), lr=run.config["lr"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if config.lr_schedule:
            if isinstance(config.lr_schedule_step_size, int):
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=run.config["lr_schedule_step_size"],
                    gamma=run.config["lr_schedule_gamma"],
                )
            else:
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[
                        int(s) for s in run.config["lr_schedule_step_size"].split("-")
                    ],
                    gamma=run.config["lr_schedule_gamma"],
                )

            if checkpoint["lr_scheduler_state_dict"] is not None:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    logger.info(
        f"Number of parameters that requires gradient: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    logger.info(
        f"Number of parameters that are being optimized: {sum(p.numel() for p in policy_model.parameters() if p.requires_grad)}"
    )

    if rank == 0 and run is not None:

        run.watch(policy_model)
    # ---------------------------------------------------------------------------- #

    # ---------------------------------------------------------------------------- #
    #                                    Trainer                                   #
    # ---------------------------------------------------------------------------- #
    trainer = Trainer(
        model,
        dataloader_train,
        dataloader_val,
        optimizer,
        rank,
        cost,
        cost_weight=config.cost_weight,
        num_time_step_previous=config.num_time_step_previous,
        num_time_step_future=config.num_time_step_future,
        binary_occupancy=config.binary_occupancy,
        num_epochs=config.num_epochs,
        current_epoch=checkpoint["epoch"] + 1 if config.resume else 0,
        lr_scheduler=lr_scheduler if config.lr_schedule else None,
        gradient_clip_type=config.gradient_clip_type,
        gradient_clip_value=config.gradient_clip_value,
        save_path=config.pretrained_model_path,
        train_step=checkpoint["train_step"] if config.resume else 0,
        val_step=checkpoint["val_step"] if config.resume else 0,
        debug_render=config.debug_render,
        save_interval=config.save_interval if rank == 0 else -1,
    )

    logger.info("Training started!")
    trainer.learn(run if rank == 0 else None)
    destroy_process_group()


if __name__ == "__main__":

    checkpoint_path = Path("pretrained_models")

    date_ = Path(datetime.today().strftime("%Y-%m-%d"))
    time_ = Path(datetime.today().strftime("%H-%M-%S"))

    checkpoint_path = checkpoint_path / date_ / time_
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Collect data from the CARLA simulator"
    )

    parser.add_argument("--seed", type=int, default=42)

    # TRAINING PARAMETERS
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--data_path_train",
        type=str,
        default="data/ground_truth_bev_model_train_data_10Hz_multichannel_bev_special_seed_33",
    )
    parser.add_argument(
        "--data_path_val",
        type=str,
        default="data/ground_truth_bev_model_train_data_10Hz_multichannel_bev_special_seed_33",
    )
    parser.add_argument("--pretrained_model_path", type=str, default=checkpoint_path)
    parser.add_argument(
        "--resume", type=lambda x: (str(x).lower() == "true"), default=False
    )
    parser.add_argument("--resume_checkpoint_number", type=int, default=14)
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--master_port", type=str, default="12356")
    parser.add_argument(
        "--lr_schedule", type=lambda x: (str(x).lower() == "true"), default=False
    )
    parser.add_argument("--lr_schedule_step_size", default=5)
    parser.add_argument("--lr_schedule_gamma", type=float, default=0.5)
    parser.add_argument("--gradient_clip_type", type=str, default="norm")
    parser.add_argument("--gradient_clip_value", type=float, default=1)
    parser.add_argument("--num_time_step_previous", type=int, default=-1)
    parser.add_argument("--num_time_step_future", type=int, default=-1)
    parser.add_argument(
        "--binary_occupancy", type=lambda x: (str(x).lower() == "true"), default=False
    )
    parser.add_argument("--dataset_dilation", type=int, default=2)
    parser.add_argument(
        "--debug_render", type=lambda x: (str(x).lower() == "true"), default=True
    )
    parser.add_argument("--save_interval", type=int, default=100)

    # POLICY MODEL PARAMETERS
    parser.add_argument("--input_ego_location", type=int, default=0)
    parser.add_argument("--input_ego_yaw", type=int, default=0)
    parser.add_argument("--input_ego_speed", type=int, default=1)
    parser.add_argument(
        "--delta_target", type=lambda x: (str(x).lower() == "true"), default=True
    )

    parser.add_argument("--occupancy_size", type=int, default=8)
    parser.add_argument("--action_size", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    # COST WEIGHTS
    parser.add_argument("--road_cost_weight", type=float, default=0.0)
    parser.add_argument("--road_on_cost_weight", type=float, default=0.0)
    parser.add_argument("--road_off_cost_weight", type=float, default=0.01)
    parser.add_argument("--road_red_yellow_cost_weight", type=float, default=0.01)
    parser.add_argument("--road_green_cost_weight", type=float, default=-0.01)
    parser.add_argument("--vehicle_cost_weight", type=float, default=0.01)
    parser.add_argument("--lane_cost_weight", type=float, default=0.01)
    parser.add_argument("--offroad_cost_weight", type=float, default=0.01)
    parser.add_argument("--action_mse_weight", type=float, default=1.0)
    parser.add_argument("--action_jerk_weight", type=float, default=0.0)
    parser.add_argument("--target_progress_weight", type=float, default=1.0)
    parser.add_argument("--target_remainder_weight", type=float, default=1.0)
    parser.add_argument("--ego_state_mse_weight", type=float, default=0.0)
    parser.add_argument("--world_state_mse_weight", type=float, default=0.0)
    # WANDB RELATED PARAMETERS
    parser.add_argument(
        "--wandb", type=lambda x: (str(x).lower() == "true"), default=True
    )
    parser.add_argument("--wandb_project", type=str, default="mbl")
    parser.add_argument(
        "--wandb_group",
        type=str,
        default="dfm-km-cp-5Hz-extended-extended-bev-toy-experiments",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default="policy+bc(continuous_occupancy)(target_difference_no_rotation)",
    )
    parser.add_argument("--wandb_id", type=str, default=None)

    parser.add_argument(
        "--ego_forward_model_path",
        type=str,
        default="pretrained_models/2022-09-30/17-49-06/ego_model_new.pt",
        help="Path to the forward model of the ego vehicle",
    )
    parser.add_argument(
        "--world_forward_model_wandb_link", type=str, default="vaydingul/mbl/31mxv8ub"
    )

    parser.add_argument("--world_forward_model_checkpoint_number", type=int, default=47)

    config = parser.parse_args()

    config.cost_weight = {k: v for (k, v) in vars(config).items() if "weight" in k}

    pprint(vars(config), depth=2)

    run = create_wandb_run(config)

    mp.spawn(main, args=(config.num_gpu, run, config), nprocs=config.num_gpu)

    run.finish()
