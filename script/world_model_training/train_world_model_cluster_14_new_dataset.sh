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
	--batch_size=60 \
	--num_workers=10 \
	--data_path_train="/kuacc/users/vaydingul20/ground_truth_bev_model_train_data_4_town_02/" \
	--data_path_val="/kuacc/users/vaydingul20/ground_truth_bev_model_val_data_4_town_02/" \
	--resume=false \
	--multi_gpu=true \
	--num_time_step_previous=20 \
	--num_time_step_future=30 \
	--reconstruction_loss="mse_loss" \
	--logvar_clip=false \
	--lr_schedule=false \
	--wandb=true \
	--wandb_project="mbl" \
	--wandb_group="world-forward-model-multi-step" \
	--wandb_name="20-30-mse_loss-gradient_clip_norm_0.3_new_dataset" \
	