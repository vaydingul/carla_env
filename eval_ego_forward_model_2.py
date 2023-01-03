import argparse

import logging

import torch
from torch.utils.data import DataLoader, Subset
from carla_env.dataset.instance import InstanceDataset
from carla_env.models.dynamic.vehicle import KinematicBicycleModel
from carla_env.evaluator.ego_model import Evaluator
from utils.train_utils import (get_device, seed_everything)
from utils.model_utils import (fetch_checkpoint_from_wandb_link)
import wandb
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d ==> %(message)s")


def main(config):

    seed_everything(config.seed)
    device = get_device()

    run = wandb.Api().run(config.ego_forward_model_wandb_link)
    checkpoint = fetch_checkpoint_from_wandb_link(
        wandb_link=config.ego_forward_model_wandb_link,
        checkpoint_number=config.ego_forward_model_checkpoint_number)
    model = KinematicBicycleModel.load_model_from_wandb_run(
        run=run, checkpoint=checkpoint, device=device)
    model.to(device).eval()

    # Create dataset and its loader
    data_path_test = config.data_path_test
    dataset_test = InstanceDataset(
        data_path=data_path_test,
        sequence_length=run.config["num_time_step_previous"] +
        run.config["num_time_step_future"],
        read_keys=["ego"],
        dilation=run.config["dataset_dilation"])

    logger.info(f"Test dataset size: {len(dataset_test)}")

    dataloader_test = DataLoader(
        dataset=Subset(
            dataset_test,
            range(
                0,
                len(dataset_test),
                (run.config["num_time_step_previous"] +
                 run.config["num_time_step_future"]) *
                run.config["dataset_dilation"])),
        batch_size=3,
        shuffle=False,
        num_workers=0)

    # dataloader_test = DataLoader(
    #     dataset=dataset_test,
    #     batch_size=3,
    #     shuffle=False,
    #     num_workers=0)

    evaluator = Evaluator(
        model=model,
        dataloader=dataloader_test,
        device=device,
        sequence_length=run.config["num_time_step_previous"] +
        run.config["num_time_step_future"],
        save_path=f"{config.save_path}")

    evaluator.evaluate(render=False, save=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data_path_test",
        type=str,
        default="/home/vaydingul/Documents/Codes/carla_env/data/kinematic_model_val_data_10Hz/")
    parser.add_argument(
        "--save_path",
        type=str,
        default="figures/ego_forward_model_evaluation_extensive/TRIAL3/")
    parser.add_argument("--num_time_step_previous", type=int, default=1)
    parser.add_argument("--num_time_step_future", type=int, default=10)
    parser.add_argument("--ego_forward_model_wandb_link", type=str,
                        default="vaydingul/mbl/ssifa1go")
    parser.add_argument(
        "--ego_forward_model_checkpoint_number",
        type=int,
        default=299)
    config = parser.parse_args()

    main(config)
