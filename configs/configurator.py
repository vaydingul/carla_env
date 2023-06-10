import yaml
import os
import argparse
from utilities.config_utils import parse_yml
from tqdm import tqdm
from copy import deepcopy


def changed(config):
    """
    It is a custom function to apply on the config file.
    Since it can be different for every task,
    it is defined in the task folder.
    """

    for file in os.listdir("leaderboard/data/routes_testing/"):
        if file.endswith(".xml"):
            config_ = deepcopy(config)


            config_["leaderboard"]["routes"] = os.path.join(
                "leaderboard/data/routes_testing/", file
            )
            
            config_["leaderboard"]["checkpoint"] = f'routes_testing_mpc/{file.replace(".xml", ".json")}'
            
            config_["environment"]["renderer"]["save_path"] = os.path.join(
                config["environment"]["renderer"]["save_path"], file.replace(".xml", "")
            )
            yield config_


def main(args):
    config = parse_yml(args.config_path)

    for i, changed_config in tqdm(enumerate(changed(config))):
		

        path = os.path.join(os.path.dirname(args.config_path), args.save_path, f"config_{i}.yml")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(changed_config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        "-c",
        default="configs/mpc_with_external_agent/roach/evaluation/5Hz/config_roach_main.yml",
        help="Path to the configuration file (default: configs/config.yml)",
    )

    parser.add_argument(
        "--save_path",
        "-s",
        default="roach_ppo+beta_enumerated",
    )

    args = parser.parse_args()

    main(args)
