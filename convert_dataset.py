import logging
import argparse
from pathlib import Path
import tqdm
import torch


from utilities.config_utils import parse_yml
from utilities.log_utils import get_logger, configure_logger, pretty_print_config
from utilities.factory import *


def main(config):

	# ---------------------------------------------------------------------------- #
	#                                    LOGGER                                    #
	# ---------------------------------------------------------------------------- #
	logger = get_logger(__name__)
	configure_logger(__name__, log_path=config["log_path"], log_level=logging.INFO)
	pretty_print_config(logger, config)

	# ---------------------------------------------------------------------------- #
	#                                 DATASET CLASS                                #
	# ---------------------------------------------------------------------------- #
	for dataset in config["datasets"]:

		dataset_class = dataset_factory(config, dataset["type"])

		dataset_instance = dataset_class(
			config=dataset["config"],
		)

		destination_path = Path(dataset["destination_path"])
		destination_path.mkdir(parents=True, exist_ok=True)
		for k in tqdm.trange(len(dataset_instance)):

			data = dataset_instance[k]

			torch.save(data, destination_path / f"{k}.pth")


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--config_path",
		type=str,
		default="/home/volkan/Documents/Codes/carla_env_modified/configs/dataset_conversion/config.yml",
	)

	args = parser.parse_args()

	config = parse_yml(args.config_path)

	config["config_path"] = args.config_path

	main(config)
