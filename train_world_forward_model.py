import argparse
from datetime import datetime
from pathlib import Path
import logging
import wandb
from typing import Union
import torch
from torch.utils.data import DataLoader
from carla_env.dataset.instance import InstanceDataset
from carla_env.models.world.world import WorldBEVModel
from carla_env.trainer.world_model import Trainer
from utils.model_utils import (
    fetch_model_from_wandb_link,
    fetch_model_from_wandb_run)

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
        num_workers=config.num_workers)
    world_model_dataloader_val = DataLoader(
        world_model_dataset_val,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers)

    # Setup the wandb
    if config.wandb:

        run = wandb.init(
            id=config.wandb_id if config.wandb_id is not None else wandb.util.generate_id(),
            project=config.wandb_project,
            group=config.wandb_group,
            name=config.wandb_name,
            resume=config.wandb_resume,
            config=config)

        if config.wandb_id is None:
            run.config["wandb_id"] = run.id

        run.define_metric("train/step")
        run.define_metric("val/step")
        run.define_metric(name="train/*", step_metric="train/step")
        run.define_metric(name="val/*", step_metric="val/step")

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

        model_file, run = fetch_model_from_wandb_run(
            wandb_link=config.wandb_link)
        world_bev_model = WorldBEVModel(
            input_shape=run.config["input_shape"],
            hidden_channel=run.config["hidden_channel"],
            output_channel=run.config["output_channel"],
            num_encoder_layer=run.config["num_encoder_layer"],
            num_probabilistic_encoder_layer=run.config[
                "num_probabilistic_encoder_layer"],
            num_time_step=run.config["num_time_step_previous"] + 1,
            dropout=run.config["dropout"])
        world_bev_model.load_state_dict(torch.load(model_file.name))

    logger.info(
        f"Number of parameters: {sum(p.numel() for p in world_bev_model.parameters() if p.requires_grad)}")

    world_model_optimizer = torch.optim.Adam(
        world_bev_model.parameters(), lr=config.lr)

    world_model_device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    world_model_trainer = Trainer(
        world_bev_model,
        world_model_dataloader_train,
        world_model_dataloader_val,
        world_model_optimizer,
        world_model_device,
        num_epochs=config.num_epochs,
        reconstruction_loss=config.reconstruction_loss,
        save_path=config.pretrained_model_path,
        train_step=run.summary["train/step"] if config.resume else 0,
        val_step=run.summary["val/step"] if config.resume else 0)

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

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_path_train", type=str,
                        default="data/ground_truth_bev_model_train_data/")
    parser.add_argument("--data_path_val", type=str,
                        default="data/ground_truth_bev_model_train_data/")
    parser.add_argument("--pretrained_model_path",
                        type=str, default=checkpoint_path)
    parser.add_argument("--resume", type=bool, default=True)

    # MODEL PARAMETERS
    parser.add_argument("--input_shape", type=list, default=[7, 192, 192])
    parser.add_argument("--hidden_channel", type=int, default=256)
    parser.add_argument("--hidden_channel", type=int, default=256)
    parser.add_argument("--output_channel", type=int, default=512)
    parser.add_argument("--num_encoder_layer", type=int, default=4)
    parser.add_argument(
        "--num_probabilistic_encoder_layer",
        type=int,
        default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_time_step_previous", type=int, default=10)
    parser.add_argument("--num_time_step_future", type=int, default=10)
    parser.add_argument("--reconstruction_loss", type=str, default="mse_loss")

    # WANDB RELATED PARAMETERS
    parser.add_argument("--wandb", type=bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="mbl")
    parser.add_argument(
        "--wandb_group",
        type=str,
        default="world-forward-model-multi-step")
    parser.add_argument("--wandb_name", type=str, default="trial1")

    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument(
        "--wandb_resume", type=Union[str, bool], default="allow")

    config = parser.parse_args()

    main(config)
