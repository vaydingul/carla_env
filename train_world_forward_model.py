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

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d ==> %(message)s")


def main(config):

    data_path_train = config.data_path_train
    data_path_val = config.data_path_val

    world_model_dataset_train = InstanceDataset(
        data_path=data_path_train,
        sequence_length=config.num_time_step)
    world_model_dataset_val = InstanceDataset(
        data_path=data_path_val,
        sequence_length=config.num_time_step)

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

    world_bev_model = WorldBEVModel(
        input_shape=[
            7,
            192,
            192],
        num_time_step=config.num_time_step)

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
        save_path=config.pretrained_model_path)

    if config.wandb:
        run = wandb.init(project="mbl", group="world-forward-model",
                         name="model", config=config)
        run.define_metric("train/step")
        run.define_metric("val/step")
        run.define_metric(name="train/*", step_metric="train/step")
        run.define_metric(name="val/*", step_metric="val/step")

    else:

        run = None

    logger.info("Training started!")
    world_model_trainer.learn(run)

    world_bev_model.to("cpu")

    run.finish()


if __name__ == "__main__":

    checkpoint_path = Path("pretrained_models")

    date_ = Path(datetime.today().strftime('%Y-%m-%d'))
    time_ = Path(datetime.today().strftime('%H-%M-%S'))

    checkpoint_path = checkpoint_path / date_ / time_
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=90)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_time_step", type=int, default=11)
    parser.add_argument("--reconstruction_loss", type=str, default="mse_loss")
    parser.add_argument("--data_path_train", type=str,
                        default="data/ground_truth_bev_model_train_data/")
    parser.add_argument("--data_path_val", type=str,
                        default="data/ground_truth_bev_model_train_data/")
    parser.add_argument("--pretrained_model_path",
                        type=str, default=checkpoint_path)
    parser.add_argument("--wandb", type=bool, default=True)
    config = parser.parse_args()

    config.wandb = False

    main(config)
