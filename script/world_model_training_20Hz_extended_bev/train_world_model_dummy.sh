#!/usr/bin/env bash

echo "World Model Training with multiple GPU!"

# Run the world model
python3 train_world_forward_model_ddp.py \
	--seed=42 \
	--lr=1e-4 \
	--num_epochs=100 \
	--batch_size=50 \
	--num_workers=4 \
	--data_path_train="data/ground_truth_bev_model_train_data_10Hz_multichannel_bev/" \
	--data_path_val="data/ground_truth_bev_model_val_data_10Hz_multichannel_bev/" \
	--resume=false \
	--num_gpu=1 \
	--master_port="12979" \
	--save_every=5 \
	--input_shape="8-192-192"\
	--latent_size=128 \
	--hidden_channel=256 \
	--output_channel=256 \
	--num_encoder_layer=6\
	--num_probabilistic_encoder_layer=1 \
	--dropout=0.1 \
	--num_time_step_previous=5 \
	--num_time_step_future=10 \
	--dataset_dilation=2 \
	--reconstruction_loss="binary_cross_entropy" \
	--bev_channel_weights="1,1,1,1,1,5,5,1" \
	--weighted_sampling=true \
	--logvar_clip=false \
	--lr_schedule=false \
	--wandb=false \
	--wandb_project="mbl" \
	--wandb_group="world-forward-model-multi-step-5Hz-extended-bev" \
	--wandb_name="5-10-binary_cross_entropy-weighted_loss-weighted_sampling" \
	
