#!/usr/bin/env bash

echo "World Model Training!"

# Conda activation
module load anaconda/2022.05
source activate carla

echo "Conda environment is activated"


# Run the world model
python3 train_world_forward_model.py \
	--lr=1e-4 \
	--num_epochs=50 \
	--batch_size=100 \
	--data_path_train="/kuacc/users/vaydingul20/ground_truth_bev_model_train_data/" \
	--data_path_val="/kuacc/users/vaydingul20/ground_truth_bev_model_val_data/" \
	--resume=false \
	--num_time_step_previous=10 \
	--num_time_step_future=10 \
	--reconstruction_loss="binary_cross_entropy" \
	--wandb=true \
	--wandb_project="mbl" \
	--wandb_group="world-forward-model-multi-step" \
	--wandb_name="10-10-binary_cross_entropy-gradient_clip" \
