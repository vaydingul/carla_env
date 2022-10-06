#!/usr/bin/env bash

# Run the ego model
python3 eval_ego_forward_model.py \
	--model_path="pretrained_models/2022-09-30/17-49-06/ego_model_new.pt" \
	--evaluation_data_folder="./data/kinematic_model_data_test_3" \
	--save_dir="train_old_dataset_test_old_dataset" \
	--kinematic_model="v2"

# python3 eval_ego_forward_model.py \
# --model_path="pretrained_models/2022-09-27/16-21-41/ego_model_new.pt" \
# --evaluation_data_folder="./data/kinematic_model_data_val" \
# --save_dir="train_old_dataset_test_old_dataset" \
# --kinematic_model="v1"

# python3 eval_ego_forward_model.py \
# --model_path="pretrained_models/2022-09-27/16-21-41/ego_model_new.pt" \
# --evaluation_data_folder="data/kinematic_model_data_val_2/" \
# --save_dir="train_old_dataset_test_new_dataset" \
# --kinematic_model="v1"

# python3 eval_ego_forward_model.py \
# --model_path="pretrained_models/2022-09-28/03-24-39/ego_model_new.pt" \
# --evaluation_data_folder="data/kinematic_model_data_val/" \
# --save_dir="train_new_dataset_test_old_dataset" \
# --kinematic_model="v1"

# python3 eval_ego_forward_model.py \
# --model_path="pretrained_models/2022-09-28/03-24-39/ego_model_new.pt" \
# --evaluation_data_folder="data/kinematic_model_data_val_2/" \
# --save_dir="train_new_dataset_test_new_dataset" \
# --kinematic_model="v1"

# python3 eval_ego_forward_model.py \
# --evaluation_data_folder="data/kinematic_model_data_val/" \
# --save_dir="WoR_test_old_dataset" \
# --kinematic_model="WoR"

# python3 eval_ego_forward_model.py \
# --evaluation_data_folder="data/kinematic_model_data_val_2/" \
# --save_dir="WoR_test_new_dataset" \
# --kinematic_model="WoR"
