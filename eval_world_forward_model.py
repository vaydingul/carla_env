import argparse
from datetime import datetime
from pathlib import Path
import logging
import wandb

import torch
from torch.utils.data import DataLoader, Subset
from carla_env.dataset.instance import InstanceDataset
from carla_env.models.world.world import WorldBEVModel
from carla_env.evaluator.world_model import Evaluator
from utils.model_utils import fetch_checkpoint_from_wandb_run, fetch_checkpoint_from_wandb_link

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d ==> %(message)s")


def main(config):

    # Load the pretrained world_bev_model

    if config.wandb_link:
        logger.info(
            f"Fetching checkpoint from wandb link: {config.wandb_link}")
        run = wandb.Api().run(config.wandb_link)
        checkpoint = fetch_checkpoint_from_wandb_link(
            config.wandb_link, config.checkpoint_number)

    else:
        logger.info(
            f"Downloading world_bev_model from wandb run {config.wandb_project}-{config.wandb_group}")
        run = wandb.init(
            project=config.wandb_project,
            group=config.wandb_group,
            id=config.wandb_id,
            resume="allow")
        checkpoint = fetch_checkpoint_from_wandb_run(
            run, config.checkpoint_number)

    world_model_device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info(f"Checkpoint downloaded: {checkpoint.name}")
    checkpoint = torch.load(
        checkpoint.name,
        map_location=world_model_device)
    world_bev_model = WorldBEVModel(
        input_shape=run.config["input_shape"],
        hidden_channel=run.config["hidden_channel"],
        output_channel=run.config["output_channel"],
        num_encoder_layer=run.config["num_encoder_layer"],
        num_probabilistic_encoder_layer=run.config["num_probabilistic_encoder_layer"],
        num_time_step=run.config["num_time_step_previous"] + 1,
        dropout=run.config["dropout"])
    world_bev_model.load_state_dict(checkpoint["model_state_dict"])
    world_bev_model.eval()

    # Create dataset and its loader
    data_path_test = config.data_path_test
    world_model_dataset_test = InstanceDataset(
        data_path=data_path_test,
        sequence_length=run.config["num_time_step_previous"] +
        config.num_time_step_predict,
        dilation=run.config["dataset_dilation"] if "dataset_dilation" in run.config.keys() else 1,
        read_keys=["bev_world"],
        bev_agent_channel=3,
        bev_vehicle_channel=2,
        bev_selected_channels=[0, 1, 2, 4, 5, 6, 7],
        bev_calculate_offroad=True)

    logger.info(f"Test dataset size: {len(world_model_dataset_test)}")

    world_model_dataloader_test = DataLoader(
        dataset=world_model_dataset_test if config.test_set_step == 1 else Subset(
            world_model_dataset_test, range(
                0, len(world_model_dataset_test), config.test_set_step)), batch_size=20)

    world_model_device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    loss_function_ = run.config['reconstruction_loss'] if 'reconstruction_loss' in run.config else 'cross_entropy'

    evaluator = Evaluator(
        model=world_bev_model,
        dataloader=world_model_dataloader_test,
        device=world_model_device,
        report_iou=config.report_iou,
        num_time_step_previous=run.config["num_time_step_previous"],
        num_time_step_predict=config.num_time_step_predict,
        threshold=config.threshold,
        save_path=f"{config.save_path}/{run.config['num_time_step_previous']}-{run.config['num_time_step_future']}-{run.config['reconstruction_loss']}-{config.threshold}-{config.checkpoint_number}")

    evaluator.evaluate(render=False, save=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path_test", type=str,
                        default="data/ground_truth_bev_model_test_data/")
    parser.add_argument("--test_set_step", type=int, default=10)
    parser.add_argument(
        "--save_path",
        type=str,
        default="figures/world_forward_model_evaluation/")
    parser.add_argument("--report_iou", type=lambda x: (
        str(x).lower() == 'true'), default=True)
    parser.add_argument(
        "--wandb_link",
        type=str,
        default="vaydingul/mbl/203kw46a")
    parser.add_argument("--wandb_project",
                        type=str,
                        default="mbl")
    parser.add_argument("--wandb_group",
                        type=str,
                        default="world-forward-model-multi-step")
    parser.add_argument("--wandb_id",
                        type=str,
                        default="3aqhglkb")
    parser.add_argument("--checkpoint_number", type=int, default=4)
    parser.add_argument("--num_time_step_predict", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.25)

    config = parser.parse_args()

    main(config)
