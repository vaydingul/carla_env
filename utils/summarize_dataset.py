
from carla_env.dataset.instance import InstanceDataset
import torch
from torch.utils.data import DataLoader
import logging
import argparse
import numpy as np
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
from carla_env.bev import BirdViewProducer
import os
logger = logging.getLogger(__name__)
logging.basicConfig(
	level=logging.INFO,
	datefmt='%Y-%m-%d %H:%M:%S',
	format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d ==> %(message)s")


def main(config):

	# Create dataset and its loader
	data_path_test = config.dataset_path
	dataset = InstanceDataset(
		data_path=data_path_test,
		sequence_length=1,
		read_keys=["navigation"])

	logger.info(f"Dataset size: {len(dataset)}")

	navigational_command_list = np.zeros((6, ))
	for i in range(len(dataset)):
		# Get data
		data = dataset[i]
		navigational_command = data["navigation"]["command"]
		navigational_command_list[int(navigational_command) - 1] += 1
		

	print(navigational_command_list)		


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_path", type=str,
						default="data/ground_truth_bev_model_test_data/")
	
	config = parser.parse_args()

	main(config)
