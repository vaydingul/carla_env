#!/usr/bin/env bash

echo "World Model Training with multiple GPU!"

# Conda activation
module load anaconda/2022.05
source activate carla

echo "Conda environment is activated"


# Run the world model
python3 train_world_forward_model_ddp.py \
	--seed=42 \
	--lr=1e-4 \
	--num_epochs=100 \
	--batch_size=20 \
	--num_workers=4 \
	--data_path_train="/kuacc/users/vaydingul20/ground_truth_bev_model_train_data_10Hz/" \
	--data_path_val="/kuacc/users/vaydingul20/ground_truth_bev_model_val_data_10Hz/" \
	--resume=true \
	--resume_checkpoint_number=14 \
	--num_gpu=5 \
	--master_port="12555" \
	--save_every=5 \
	--num_time_step_previous=20 \
	--num_time_step_future=30 \
	--dataset_dilation=2 \
	--reconstruction_loss="mse_loss" \
	--logvar_clip=false \
	--lr_schedule=false \
	--wandb=true \
	--wandb_id="2xpqn2jg" \
	--wandb_project="mbl" \
	--wandb_group="world-forward-model-multi-step-5Hz" \
	--wandb_name="20-30-mse_loss-gradient_clip_norm_0.3_5Hz" \
	
