#!/usr/bin/env bash

echo "DFM with KM CP Training!"

# Run the world model
python3 train_dfm_km_cp_extended_bev_gt_bev_encoded_policy_fused_20Hz.py \
	--lr=1e-4 \
	--num_epochs=50 \
	--batch_size=30 \
	--num_workers=5 \
	--data_path_train="./data/ground_truth_bev_model_dummy_data_20Hz_multichannel_bev_dense_traffic/" \
	--data_path_val="./data/ground_truth_bev_model_dummy_data_20Hz_multichannel_bev_dense_traffic/" \
	--resume=false \
	--num_gpu=1 \
	--master_port="12355" \
	--lr_schedule=false \
	--gradient_clip_type="norm" \
	--gradient_clip_value=1.0 \
	--binary_occupancy=false \
	--num_time_step_future=10 \
	--dataset_dilation=1 \
	--debug_render=true \
	--save_interval=5 \
	--input_ego_location=0 \
	--input_ego_yaw=0 \
	--input_ego_speed=1 \
	--delta_target=true \
	--occupancy_size=8 \
	--action_size=2 \
	--hidden_size=256 \
	--num_layer=3 \
	--dropout=0.1 \
	--road_cost_weight=0.0 \
	--road_on_cost_weight=0.0 \
	--road_off_cost_weight=0.01 \
	--road_red_yellow_cost_weight=0.01 \
	--road_green_cost_weight=-0.01 \
	--lane_cost_weight=0.01 \
	--vehicle_cost_weight=0.01 \
	--offroad_cost_weight=0.01 \
	--action_mse_weight=1.0 \
	--action_jerk_weight=0.0 \
	--target_progress_weight=-1.0 \
	--target_remainder_weight=1.0 \
	--ego_state_mse_weight=0.0 \
	--wandb=true \
	--wandb_project="mbl" \
	--wandb_group="dfm-km-cp-20Hz-extended-bev-toy-experiments" \
	--wandb_name="policy+bc+target(bev_encoded_policy_fused)(new_cost_parameters)" \
	--ego_forward_model_path="pretrained_models/2022-09-30/17-49-06/ego_model_new.pt" \
	--world_forward_model_wandb_link="vaydingul/mbl/kesa7b2p" \
	--world_forward_model_checkpoint_number=49
