import argparse
from datetime import datetime
from pathlib import Path
import logging
import wandb
import torch
from torch.utils.data import DataLoader
from carla_env.dataset.instance import InstanceDataset
from carla_env.models.world.world import WorldBEVModel
from carla_env.trainer.world_model import Trainer
from utils.model_utils import (
    fetch_checkpoint_from_wandb_link,
    fetch_checkpoint_from_wandb_run)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d ==> %(message)s")


def main(config):
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
        drop_last=True)
    world_model_dataloader_val = DataLoader(
        world_model_dataset_val,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=True)

    world_model_device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup the wandb
    if config.wandb:

        if not config.resume:
            run = wandb.init(
                id=wandb.util.generate_id(),
                project=config.wandb_project,
                group=config.wandb_group,
                name=config.wandb_name,
                resume="allow",
                config=config)

            if config.wandb_id is None:
                run.config.update({"wandb_id": run.id}, allow_val_change=True)

            run.define_metric("train/step")
            run.define_metric("val/step")
            run.define_metric(name="train/*", step_metric="train/step")
            run.define_metric(name="val/*", step_metric="val/step")

        else:

            run = wandb.init(
                project=config.wandb_project,
                group=config.wandb_group,
                id=config.wandb_id,
                resume="allow")

    else:

        run = None

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
        checkpoint = torch.load(
            checkpoint.name,
            map_location=world_model_device)
        world_bev_model = WorldBEVModel(
            input_shape=run.config["input_shape"],
            hidden_channel=run.config["hidden_channel"],
            output_channel=run.config["output_channel"],
            num_encoder_layer=run.config["num_encoder_layer"],
            num_probabilistic_encoder_layer=run.config[
                "num_probabilistic_encoder_layer"],
            num_time_step=run.config["num_time_step_previous"] + 1,
            dropout=run.config["dropout"])
        world_bev_model.load_state_dict(checkpoint["model_state_dict"])

    world_bev_model.to(world_model_device)

    logger.info(
        f"Number of parameters: {sum(p.numel() for p in world_bev_model.parameters() if p.requires_grad)}")

    if not config.resume:
        world_model_optimizer = torch.optim.Adam(
            world_bev_model.parameters(), lr=config.lr)
    else:
        world_model_optimizer = torch.optim.Adam(
            world_bev_model.parameters(), lr=run.config["lr"])
        world_model_optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"])

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
    parser.add_argument("--wandb_name", type=str, default="trial1")

    parser.add_argument("--wandb_id", type=str, default=None)

    config = parser.parse_args()

    main(config)
