#!/usr/bin/env bash

echo "DFM with KM CP Training!"

# Conda activation
module load anaconda/2022.05
source activate carla

echo "Conda environment is activated"



# Run the world model
python3 train_dfm_km_cp_extended_bev_gt_bev.py \
	--lr=1e-4 \
	--num_epochs=50 \
	--batch_size=95 \
	--num_workers=4 \
	--data_path_train="/kuacc/users/vaydingul20/ground_truth_bev_model_train_data_10Hz_multichannel_bev_dense_traffic/" \
	--data_path_val="/kuacc/users/vaydingul20/ground_truth_bev_model_val_data_10Hz_multichannel_bev_dense_traffic/" \
	--resume=false \
	--num_gpu=4 \
	--master_port="12355" \
	--lr_schedule=false \
	--gradient_clip_type="norm" \
	--gradient_clip_value=1.0 \
	--dataset_dilation=2 \
	--debug_render=true \
	--save_interval=5 \
	--input_ego_location=0 \
	--input_ego_yaw=0 \
	--input_ego_speed=1 \
	--delta_target=true \
	--single_world_state_input=false \
	--occupancy_size=8 \
	--action_size=2 \
	--hidden_size=256 \
	--num_layer=6 \
	--dropout=0.0 \
	--road_cost_weight=0.0 \
	--road_on_cost_weight=0.0 \
	--road_off_cost_weight=0.1 \
	--road_red_yellow_cost_weight=0.1 \
	--road_green_cost_weight=-0.1 \
	--lane_cost_weight=0.1 \
	--vehicle_cost_weight=0.1 \
	--offroad_cost_weight=0.1 \
	--action_mse_weight=1.0 \
	--action_jerk_weight=0.0 \
	--target_progress_weight=0.0 \
	--target_remainder_weight=0.0 \
	--ego_state_mse_weight=0.0 \
	--wandb=true \
	--wandb_project="mbl" \
	--wandb_group="dfm-km-cp-5Hz-extended-extended-bev" \
	--wandb_name="policy+bc(continuous_occupancy)(target_difference_no_rotation)" \
	--ego_forward_model_wandb_link="vaydingul/mbl/ssifa1go" \
	--ego_forward_model_checkpoint_number=449 \
	--world_forward_model_wandb_link="vaydingul/mbl/31mxv8ub" \
	--world_forward_model_checkpoint_number=47
