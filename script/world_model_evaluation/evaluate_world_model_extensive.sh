#!/usr/bin/env bash

#wandb_links=("vaydingul/mbl/203kw46a" "vaydingul/mbl/q4xzu1de" "vaydingul/mbl/phys7134" "vaydingul/mbl/3vbs6lik" "vaydingul/mbl/3qkyatvq" "vaydingul/mbl/1q48qjcg" "vaydingul/mbl/wvdt0p1k")
#wandb_links=("vaydingul/mbl/1gftiw9w" "vaydingul/mbl/n28cn1kw" "vaydingul/mbl/9atz96u8" "vaydingul/mbl/qeg93zho" "vaydingul/mbl/2wdavsik" "vaydingul/mbl/3frb4lzq" "vaydingul/mbl/254636mi" "vaydingul/mbl/369uz935" "vaydingul/mbl/2v1kvf81" "vaydingul/mbl/wvdt0p1k" "vaydingul/mbl/1q48qjcg" "vaydingul/mbl/3qkyatvq" "vaydingul/mbl/3vbs6lik" "vaydingul/mbl/phys7134" "vaydingul/mbl/q4xzu1de" "vaydingul/mbl/203kw46a")
wandb_links=("vaydingul/mbl/23mnzxda") 
batch_sizes=(100)
checkpoint_numbers=(95)
# checkpoint_numbers=(4 9 14 19 24 29 34)
# thresholds=(0.1 0.25 0.5)

thresholds=(0.5)
vehicle_thresholds=(0.1)

FOLDER_NAME="world_forward_model_extended_bev_evaluation_extensive_5Hz_with_metrics"

length_wandb_links=${#wandb_links[@]}
length_checkpoint_numbers=${#checkpoint_numbers[@]}
length_thresholds=${#thresholds[@]}
# Run the world model
for ((j = 0; j < $length_checkpoint_numbers; j++)); do
	for ((i = 0; i < $length_wandb_links; i++)); do
		for ((k = 0; k < $length_thresholds; k++)); do

			# echo bold red line 
			echo -e "\033[1;31m====================================================================================================\033[0m"
			echo "${wandb_links[i]} | ${checkpoint_numbers[j]} | ${thresholds[k]}"

			python3 eval_world_forward_model.py \
				--wandb_link="${wandb_links[i]}" \
				--checkpoint_number="${checkpoint_numbers[j]}" \
				--data_path_test="data/ground_truth_bev_model_test_data_10Hz_multichannel_bev/" \
				--save_path="figures/$FOLDER_NAME/$(echo ${wandb_links[i]} | cut -d "/" -f 3)/${thresholds[k]}/" \
				--test_set_step=10 \
				--batch_size="${batch_sizes[i]}" \
				--num_time_step_predict=-1 \
				--plot_prediction=true \
				--report_metrics=false \
				--metrics="iou,accuracy,precision,recall,f1" \
				--threshold="${thresholds[k]}" \
				--vehicle_threshold="${vehicle_thresholds[k]}" \
				--bev_selected_channels="0,1,2,3,4,5,6,11"

			echo -e "\033[1;31m====================================================================================================\033[0m"
		done
	done
done
