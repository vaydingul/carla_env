import argparse

import logging

import torch
from torch.utils.data import DataLoader, Subset
from carla_env.dataset.instance import InstanceDataset
from carla_env.models.dynamic.vehicle import KinematicBicycleModel, KinematicBicycleModelV2, KinematicBicycleModelWoR
from carla_env.evaluator.ego_model import Evaluator
from utils.train_utils import get_device

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d ==> %(message)s")


def main(config):

    device = get_device()

    model = KinematicBicycleModelV2(dt=1 / 20)
    model.state_dict = torch.load(config.model_path)
    model.eval()

    # Create dataset and its loader
    data_path_test = config.data_path_test
    dataset_test = InstanceDataset(
        data_path=data_path_test,
        sequence_length=10,
        read_keys=["ego"])

    logger.info(f"Test dataset size: {len(dataset_test)}")

    dataloader_test = DataLoader(
        dataset=Subset(dataset_test, range(0, len(dataset_test), 10)), batch_size=20, shuffle=False, num_workers=0)

    
    evaluator = Evaluator(
        model=model,
        dataloader=dataloader_test,
        device=device,
        save_path=f"{config.save_path}")

    evaluator.evaluate(render=False, save=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path_test", type=str,
                        default="data/ground_truth_bev_model_test_data_4_town_02/")
    parser.add_argument(
        "--model_path",
        type=str,
        default="pretrained_models/2022-09-30/17-49-06/ego_model_new.pt")
    parser.add_argument(
        "--save_path",
        type=str,
        default="figures/ego_forward_model_evaluation_extensive/TRIAL/")

    config = parser.parse_args()

    main(config)
