#!/usr/bin/env bash

echo "DFM with KM CP Training!"

# Run the world model
python3 train_dfm_km_cp_extended_bev_gt_bev.py \
	--lr=1e-4 \
	--num_epochs=1000 \
	--batch_size=70 \
	--num_workers=4 \
	--data_path_train="data/ground_truth_bev_model_train_data_10Hz_multichannel_bev_special_seed_33" \
	--data_path_val="data/ground_truth_bev_model_train_data_10Hz_multichannel_bev_special_seed_33" \
	--resume=false \
	--num_gpu=1 \
	--master_port="12355" \
	--lr_schedule=false \
	--gradient_clip_type="norm" \
	--gradient_clip_value=1.0 \
	--binary_occupancy=false \
	--num_time_step_future=10 \
	--dataset_dilation=2 \
	--debug_render=true \
	--save_interval=50 \
	--input_ego_location=0 \
	--input_ego_yaw=0 \
	--input_ego_speed=1 \
	--delta_target=true \
	--single_world_state_input=false \
	--occupancy_size=8 \
	--action_size=2 \
	--hidden_size=256 \
	--num_layer=4 \
	--dropout=0.0 \
	--road_cost_weight=0.0 \
	--road_on_cost_weight=0.0 \
	--road_off_cost_weight=0.01 \
	--road_red_yellow_cost_weight=0.01 \
	--road_green_cost_weight=-0.01 \
	--lane_cost_weight=0.01 \
	--vehicle_cost_weight=0.01 \
	--offroad_cost_weight=0.01 \
	--action_mse_weight=10.0 \
	--action_jerk_weight=0.0 \
	--target_progress_weight=-1.0 \
	--target_remainder_weight=1.0 \
	--ego_state_mse_weight=0.0 \
	--wandb=true \
	--wandb_project="mbl" \
	--wandb_group="dfm-km-cp-5Hz-extended-extended-bev-toy-experiments" \
	--wandb_name="policy+bc(continuous_occupancy)(target_difference_no_rotation)" \
	--ego_forward_model_wandb_link="vaydingul/mbl/ssifa1go" \
	--ego_forward_model_checkpoint_number=449 \
	--world_forward_model_wandb_link="vaydingul/mbl/2aed7ypg" \
	--world_forward_model_checkpoint_number=89
