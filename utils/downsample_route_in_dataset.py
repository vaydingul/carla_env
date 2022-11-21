import json
import argparse
import os
from pathlib import Path
import math


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])
                     ** 2 + (p1[2] - p2[2])**2)


def downsample_route(route, sample_factor=30):
    """
    Downsample the route by some factor.
    :param route: the trajectory , has to contain the waypoints and the road options
    :param sample_factor: Maximum distance between samples
    :return: returns the ids of the final route that can
    """

    ids_to_sample = []
    prev_option = None
    dist = 0

    for i, point in enumerate(route):
        curr_option = point[1]

        # Lane changing
        if curr_option in (5, 6):
            ids_to_sample.append(i)
            dist = 0

        # When road option changes
        elif prev_option != curr_option and prev_option not in (5,
                                                                6):
            if (i - 1) not in ids_to_sample:
                ids_to_sample.append(max(0, i - 1))
                dist = 0

        # After a certain max distance
        elif dist > sample_factor:
            ids_to_sample.append(i)
            dist = 0

        # At the end
        elif i == len(route) - 1:
            ids_to_sample.append(i)
            dist = 0

        # Compute the distance traveled
        else:
            curr_location = point[0]
            prev_location = route[i - 1][0]
            dist += distance(curr_location, prev_location)

        prev_option = curr_option

    route_downsampled = []
    last_id = 0
    for i in ids_to_sample:

        for k in range(last_id, i + 1):

            route_downsampled.append(route[i])

        last_id = k + 1

    return route_downsampled


def main(config):
    dataset_path = Path(config.dataset_path)

    for item in os.listdir(dataset_path):
        if item.startswith("episode"):

            episode_path = dataset_path / item
            episode_navigation_path = episode_path / Path("navigation")
            navigation_files = [file for file in os.listdir(
                episode_navigation_path) if file.endswith(".json")]
            navigation_files_sorted = sorted(
                navigation_files, key=lambda x: int(
                    x.split(".")[0]))

            navigation_data_list = []
            # Read the navigation file
            for navigation_file in navigation_files_sorted:
                with open(episode_navigation_path / navigation_file, "r") as f:
                    navigation_data = json.load(f)
                    navigation_data_list.append(
                        (navigation_data["waypoint"], navigation_data["command"]))

            navigation_data_downsampled_list = downsample_route(
                navigation_data_list)

            assert len(navigation_data_downsampled_list) == len(
                navigation_files_sorted)

            for k in range(len(navigation_files_sorted)):
                with open(episode_navigation_path / navigation_files_sorted[k], "w") as f:
                    json.dump(
                        obj={"command":navigation_data_downsampled_list[k][1], "waypoint":navigation_data_downsampled_list[k][0]},
                        fp=f,
                        indent=10)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Downsample route in dataset')
    parser.add_argument(
        '--dataset_path',
        type=str,
        default="/home/vaydingul/Documents/Codes/carla_env/data/ground_truth_bev_model_data_dummy_2")
    config = parser.parse_args()

    main(config)
