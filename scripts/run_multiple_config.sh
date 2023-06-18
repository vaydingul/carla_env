#!/bin/bash

# Set the directory path where the .yml files are located
directory="/home/volkan/Documents/Codes/carla_env/configs/mpc_with_external_agent/roach/evaluation/10Hz/roach_cilrs_mpc_0.01_enumerated_town_02/"

# Set the Python script to run
python_script="eval_mpc_leaderboard.py"

# Loop through each .yml file in the directory without changing the directory
for file in "$directory"/*.yml; do
	echo "Running $file"
	# Run the Python script with the .yml file as an argument
	python3 $python_script --config_path $file

	killall -9 -r CarlaUE4
	killall -9 -r python

done
