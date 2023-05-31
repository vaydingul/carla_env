import argparse
import os
import numpy as np
import json
from pathlib import Path


def main(config):
    dataset_path = Path(config.dataset_path)
    for item in os.listdir(config.dataset_path):
        if item.startswith("episode"):

            episode_path = dataset_path / item
            ego_path = episode_path / "ego"

            step_list = sorted([int(x.split(".")[0]) for x in os.listdir(ego_path)])

            ego_list = []
            for i in range(len(step_list)):

                # Read json file
                ego = json.load(open(ego_path / f"{step_list[i]}.json"))
                ego_ = {}
                for (key, value) in ego.items():

                    if value != "<<??>>":
                        ego_[key] = np.array(value)
                ego_list.append(ego_)

            ego_ = {}
            for (key, value) in ego_list[0].items():

                if key.endswith("array"):

                    ego_[key] = np.stack([ego[key] for ego in ego_list], axis=0)

            if np.diff(ego_["location_array"], axis=0).sum() == 0:
                print(f"Episode {item} is stationary!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/vaydingul/Documents/Codes/carla_env/data/ground_truth_bev_model_train_data_2/",
    )

    config = parser.parse_args()

    main(config)
