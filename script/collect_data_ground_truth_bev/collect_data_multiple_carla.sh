#!/usr/bin/env bash
# Number of CARLA instances to run
N=4
# Number of episodes to run per CARLA instance
E=10
# Folder to save the data
F="data/ground_truth_bev_model_train_data_10Hz_multichannel_bev_dense_traffic"

ports=

# Start CARLA instances
for ((i = 0; i < N; i++)); do

	# Start CARLA instance
	echo "${F}_${i}"
	echo $E
	echo $(((i+2)*1000))
	echo $((8000 + i))
	python3 collect_data_ground_truth_bev_model.py --data_save_path="${F}_${i}" --num_episodes=$E --port=$(((i+2)*1000)) --tm_port=$((8000 + i)) &

	sleep 2
	
	

done
