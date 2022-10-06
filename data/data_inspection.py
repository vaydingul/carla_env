import os
import argparse
import numpy as np


def main(config):
    for file in os.listdir(config.data_path):
        if file.endswith(".npz"):
            data = np.load(os.path.join(config.data_path, file))
            vehicle_location = data["vehicle_location"]
            vehicle_rotation = data["vehicle_rotation"]
            vehicle_velocity = data["vehicle_velocity"]
            vehicle_control = data["vehicle_control"]
            elapsed_time = data["elapsed_time"]
        print(f"File name: {file}")
        print(f"Number of samples: {vehicle_location.shape[0]}")
        print(f"Total elapsed time: {(elapsed_time - elapsed_time[0])[-1]}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Inspect dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/kinematic_model_data_train_3/",
        help="Path to the data")
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="Print statistics")
    config = parser.parse_args()

    main(config)
