import argparse
from datetime import datetime
from pathlib import Path
import logging
import wandb
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import (init_process_group, destroy_process_group)
import os

from carla_env.dataset.instance import InstanceDataset
from carla_env.models.world.world import WorldBEVModel
from carla_env.trainer.world_model_ddp import Trainer
from utils.model_utils import (
    fetch_checkpoint_from_wandb_link,
    fetch_checkpoint_from_wandb_run,
    load_world_model_from_wandb_run)
from utils.wandb_utils import (create_initial_run, create_resumed_run)
from utils.train_utils import (seed_everything, get_device)
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d ==> %(message)s")


def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
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

    ddp_setup(rank, world_size)

    seed_everything(seed=config.seed)

    # Load the dataset its loader
    world_model_dataset_train = InstanceDataset(
        data_path=config.data_path_train,
        sequence_length=config.num_time_step_previous +
        config.num_time_step_future)
    world_model_dataset_val = InstanceDataset(
        data_path=config.data_path_val,
        sequence_length=config.num_time_step_previous +
        config.num_time_step_future)

    logger.info(f"Train dataset size: {len(world_model_dataset_train)}")
    logger.info(f"Val dataset size: {len(world_model_dataset_val)}")

    world_model_dataloader_train = DataLoader(
        world_model_dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
        sampler=DistributedSampler(world_model_dataset_train))

    world_model_dataloader_val = DataLoader(
        world_model_dataset_val,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=True,
        sampler=DistributedSampler(world_model_dataset_val))

    if not config.resume:
        world_bev_model = WorldBEVModel(
            input_shape=config.input_shape,
            hidden_channel=config.hidden_channel,
            output_channel=config.output_channel,
            num_encoder_layer=config.num_encoder_layer,
            num_probabilistic_encoder_layer=config.num_probabilistic_encoder_layer,
            num_time_step=config.num_time_step_previous + 1,
            dropout=config.dropout)
    else:

        checkpoint = fetch_checkpoint_from_wandb_run(
            run=run)

        world_bev_model, _ = WorldBEVModel.load_model_from_wandb_run(
            run=run, checkpoint=checkpoint, device=rank)

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

    if rank == config.gpu_ids[0]:
        run.watch(world_bev_model)

    world_model_trainer = Trainer(
        world_bev_model,
        world_model_dataloader_train,
        world_model_dataloader_val,
        world_model_optimizer,
        rank,
        save_every=config.save_every if rank == config.gpu_ids[0] else -1,
        num_time_step_previous=config.num_time_step_previous,
        num_time_step_future=config.num_time_step_future,
        num_epochs=config.num_epochs,
        current_epoch=checkpoint["epoch"] + 1 if config.resume else 0,
        reconstruction_loss=config.reconstruction_loss,
        logvar_clip=config.logvar_clip,
        logvar_clip_min=config.logvar_clip_min,
        logvar_clip_max=config.logvar_clip_max,
        lr_scheduler=world_model_lr_scheduler if config.lr_schedule else None,
        gradient_clip_type=config.gradient_clip_type,
        gradient_clip_value=config.gradient_clip_value,
        save_path=config.pretrained_model_path,
        train_step=checkpoint["train_step"] if config.resume else 0,
        val_step=checkpoint["val_step"] if config.resume else 0)

    logger.info("Training started!")
    world_model_trainer.learn(run)

    run.finish()


if __name__ == "__main__":

    checkpoint_path = Path("pretrained_models")

    date_ = Path(datetime.today().strftime('%Y-%m-%d'))
    time_ = Path(datetime.today().strftime('%H-%M-%S'))

    checkpoint_path = checkpoint_path / date_ / time_
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)

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
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=5)
    # MODEL PARAMETERS
    parser.add_argument("--input_shape", type=list, default=[8, 192, 192])
    parser.add_argument("--hidden_channel", type=int, default=256)
    parser.add_argument("--output_channel", type=int, default=512)
    parser.add_argument("--num_encoder_layer", type=int, default=4)
    parser.add_argument(
        "--num_probabilistic_encoder_layer",
        type=int,
        default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_time_step_previous", type=int, default=10)
    parser.add_argument("--num_time_step_future", type=int, default=10)
    parser.add_argument("--reconstruction_loss", type=str, default="mse_loss")
    parser.add_argument("--logvar_clip", type=lambda x: (
        str(x).lower() == 'true'), default=True)
    parser.add_argument("--logvar_clip_min", type=float, default=-5)
    parser.add_argument("--logvar_clip_max", type=float, default=5)
    parser.add_argument("--lr_schedule", type=lambda x: (
        str(x).lower() == 'true'), default=True)
    parser.add_argument("--lr_schedule_step_size", default=5)
    parser.add_argument("--lr_schedule_gamma", type=float, default=0.5)
    parser.add_argument("--gradient_clip_type", type=str, default="norm")
    parser.add_argument("--gradient_clip_value", type=float, default=0.3)

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

    config = parser.parse_args()

    run = create_wandb_run(config)

    mp.spawn(main, args=(config.num_gpu, run, config), nprocs=config.num_gpu)