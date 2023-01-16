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
	--num_epochs=200 \
	--batch_size=140 \
	--num_workers=4 \
	--data_path_train="/kuacc/users/vaydingul20/ground_truth_bev_model_train_data_10Hz_multichannel_bev/" \
	--data_path_val="/kuacc/users/vaydingul20/ground_truth_bev_model_val_data_10Hz_multichannel_bev/" \
	--resume=true \
	--resume_checkpoint_number=99 \
	--num_gpu=8 \
	--master_port="12885" \
	--save_every=5 \
	--input_shape="8-192-192"\
	--num_time_step_previous=5 \
	--num_time_step_future=10 \
	--dataset_dilation=2 \
	--reconstruction_loss="binary_cross_entropy" \
	--bev_channel_weights="1,1,1,1,1,5,5,1" \
	--logvar_clip=false \
	--lr_schedule=false \
	--wandb=true \
	--wandb_id="178mirqe"
	--wandb_project="mbl" \
	--wandb_group="world-forward-model-multi-step-5Hz-extended-bev" \
	--wandb_name="5-10-binary_cross_entropy-weighted-gradient_clip_norm_0.3" \
	
