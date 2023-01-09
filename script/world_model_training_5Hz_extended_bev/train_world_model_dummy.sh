#!/usr/bin/env bash

echo "World Model Training with multiple GPU!"



# Run the world model
python3 train_world_forward_model_ddp.py \
	--seed=42 \
	--lr=1e-4 \
	--num_epochs=100 \
	--batch_size=1 \
	--num_workers=4 \
	--data_path_train="data/ground_truth_bev_model_dummy_data_10Hz_multichannel_bev/" \
	--data_path_val="data/ground_truth_bev_model_dummy_data_10Hz_multichannel_bev/" \
	--resume=false \
	--num_gpu=1 \
	--master_port="12555" \
	--save_every=50 \
	--input_shape="8-192-192"\
	--dropout=0.0 \
	--num_time_step_previous=5 \
	--num_time_step_future=10 \
	--dataset_dilation=2 \
	--reconstruction_loss="mse_loss" \
	--logvar_clip=false \
	--lr_schedule=false \
	--wandb=true \
	--wandb_project="mbl" \
	--wandb_group="world-forward-model-multi-step-5Hz-extended-bev" \
	--wandb_name="dummy" \
	
