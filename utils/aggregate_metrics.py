import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import os
from pathlib import Path
import argparse


def main(config):

	folder_path = Path(config.folder_path)
	df_dict = {}
	for run_folder in folder_path.iterdir():
		if run_folder.is_dir():
			checkpoint_dict = {}
			for checkpoint in (run_folder / "0.5").iterdir():
				metric_file = checkpoint / "metrics.csv"
				if metric_file.exists():
					df = pd.read_csv(metric_file)
					checkpoint_dict[int(checkpoint.name.split("-")[-1])] = df
		# Sort checkpoint keys before adding to the dictionary
		checkpoint_dict = {k: checkpoint_dict[k] for k in sorted(checkpoint_dict.keys())}
		df_dict[run_folder.name] = checkpoint_dict

	final_df = a = pd.concat({k:pd.concat(v) for (k,v) in df_dict.items()})
	final_df.to_excel(folder_path / "aggregate_metrics.xlsx")


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--folder_path",
		type=str,
		default="figures/world_forward_model_extended_bev_evaluation_extensive_5Hz_with_metrics")

	config = parser.parse_args()

	main(config)
