#!/usr/bin/env bash

#wandb_links=("vaydingul/mbl/203kw46a" "vaydingul/mbl/q4xzu1de" "vaydingul/mbl/phys7134" "vaydingul/mbl/3vbs6lik" "vaydingul/mbl/3qkyatvq" "vaydingul/mbl/1q48qjcg" "vaydingul/mbl/wvdt0p1k")
#wandb_links=("vaydingul/mbl/1gftiw9w" "vaydingul/mbl/n28cn1kw" "vaydingul/mbl/9atz96u8" "vaydingul/mbl/qeg93zho" "vaydingul/mbl/2wdavsik" "vaydingul/mbl/3frb4lzq" "vaydingul/mbl/254636mi" "vaydingul/mbl/369uz935" "vaydingul/mbl/2v1kvf81" "vaydingul/mbl/wvdt0p1k" "vaydingul/mbl/1q48qjcg" "vaydingul/mbl/3qkyatvq" "vaydingul/mbl/3vbs6lik" "vaydingul/mbl/phys7134" "vaydingul/mbl/q4xzu1de" "vaydingul/mbl/203kw46a")
wandb_links=("vaydingul/mbl/bylewhod")
checkpoint_numbers=(79 74 69 64 59 54)
# checkpoint_numbers=(4 9 14 19 24 29 34)
# thresholds=(0.1 0.25 0.5)
thresholds=(0.5)

length_wandb_links=${#wandb_links[@]}
length_checkpoint_numbers=${#checkpoint_numbers[@]}
length_thresholds=${#thresholds[@]}
# Run the world model

for ((i = 0; i < $length_wandb_links; i++)); do
	for ((k = 0; k < $length_thresholds; k++)); do
		for ((j = 0; j < $length_checkpoint_numbers; j++)); do
			echo "${wandb_links[i]} | ${checkpoint_numbers[j]} | ${thresholds[k]}"

			python3 eval_world_forward_model.py \
				--wandb_link="${wandb_links[i]}" \
				--checkpoint_number="${checkpoint_numbers[j]}" \
				--data_path_test="data/ground_truth_bev_model_test_data_4_town_02/" \
				--save_path="figures/DENEME/${wandb_links[i]}/${thresholds[k]}/" \
				--test_set_step=10 \
				--num_time_step_predict=30 \
				--report_iou=true \
				--threshold="${thresholds[k]}"
		done
	done
done
