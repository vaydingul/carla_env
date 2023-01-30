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
	--num_epochs=48 \
	--batch_size=50 \
	--num_workers=5 \
	--data_path_train="/kuacc/users/vaydingul20/ground_truth_bev_model_train_data_10Hz_multichannel_bev_dense_traffic/" \
	--data_path_val="/kuacc/users/vaydingul20/ground_truth_bev_model_val_data_10Hz_multichannel_bev_dense_traffic/" \
	--resume=false \
	--num_gpu=4 \
	--master_port="12556" \
	--save_every=6 \
	--val_every=3 \
	--input_shape="8-192-192"\
	--latent_size=64 \
	--hidden_channel=256 \
	--output_channel=512 \
	--num_encoder_layer=4\
	--num_probabilistic_encoder_layer=2 \
	--dropout=0.1 \
	--num_time_step_previous=20 \
	--num_time_step_future=10 \
	--dataset_dilation=2 \
	--reconstruction_loss="binary_cross_entropy" \
	--bev_channel_weights="1,1,1,1,1,5,5,1" \
	--weighted_sampling=true \
	--report_metrics=false \
	--logvar_clip=false \
	--lr_schedule=false \
	--wandb=true \
	--wandb_project="mbl" \
	--wandb_group="world-forward-model-multi-step-5Hz-extended-bev" \
	--wandb_name="20-10-small_latent-binary_cross_entropy-weighted_loss-weighted_sampling(new_rotation_weight)(new_dataset)" \
	
