import argparse
from datetime import datetime
from pathlib import Path
import logging
import wandb

import torch
from torch.utils.data import DataLoader
from carla_env.dataset.instance import InstanceDataset
from carla_env.models.world.world import WorldBEVModel
from carla_env.evaluator.world_model import Evaluator
from utils.model_utils import fetch_model_from_wandb

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d ==> %(message)s")


def main(config):

    # Load the pretrained model

    logger.info(f"Downloading model from wandb run {config.wandb_link}")
    (model_file, run) = fetch_model_from_wandb(wandb_link=config.wandb_link)

    model = WorldBEVModel(
        input_shape=[
            7,
            192,
            192],
        num_time_step=run.config["num_time_step"])
    model.load_state_dict(torch.load(model_file.name))
    model.eval()

    # Create dataset and its loader
    data_path_test = config.data_path_test
    world_model_dataset_test = InstanceDataset(
        data_path=data_path_test,
        sequence_length=run.config["num_time_step"])

    logger.info(f"Test dataset size: {len(world_model_dataset_test)}")

    world_model_dataloader_test = DataLoader(
        dataset=world_model_dataset_test,
        batch_size=config.num_time_step_predict)

    world_model_device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    evaluator = Evaluator(
        model=model,
        dataloader=world_model_dataloader_test,
        device=world_model_device,
        num_time_step_predict=config.num_time_step_predict,
        save_path=config.save_path)

    evaluator.evaluate(render=False, save=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path_test", type=str,
                        default="data/ground_truth_bev_model_test_data/")
    parser.add_argument(
        "--save_path",
        type=str,
        default="figures/world_forward_model_evaluation_2/")
    parser.add_argument("--wandb_link",
                        type=str,
                        default="vaydingul/mbl/14k60iqj")
    parser.add_argument("--num_time_step_predict", type=int, default=10)

    config = parser.parse_args()

    main(config)
