#!/usr/bin/env bash

echo "Ego Forward Model Training"

# Run the world model
python3 train_ego_forward_model.py \
	--seed=42 \
	--lr=1e-3 \
	--num_epochs=1000 \
	--batch_size=100 \
	--num_workers=4 \
	--data_path_train="/home/volkan/Documents/Codes/carla_env/data/kinematic_model_train_data_10Hz" \
	--data_path_val="/home/volkan/Documents/Codes/carla_env/data/kinematic_model_val_data_10Hz" \
	--num_time_step_previous=1 \
	--num_time_step_future=5 \
	--dt=0.2 \
	--dataset_dilation=2 \
	--resume=false \
	--wandb=true \
	--wandb_project="mbl" \
	--wandb_group="ego-forward-model-multi-step-5Hz" \
	--wandb_name="1-10-5Hz-complex" \