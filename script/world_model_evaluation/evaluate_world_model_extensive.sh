#!/usr/bin/env bash

#wandb_links=("vaydingul/mbl/203kw46a" "vaydingul/mbl/q4xzu1de" "vaydingul/mbl/phys7134" "vaydingul/mbl/3vbs6lik" "vaydingul/mbl/3qkyatvq" "vaydingul/mbl/1q48qjcg" "vaydingul/mbl/wvdt0p1k")
wandb_links=("vaydingul/mbl/1gftiw9w" "vaydingul/mbl/n28cn1kw")
checkpoint_numbers=(44 49)
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
				--data_path_test="data/ground_truth_bev_model_test_data/" \
				--save_path="figures/world_forward_model_evaluation_extensive/${wandb_links[i]}/${thresholds[k]}/" \
				--test_set_step=10 \
				--threshold="${thresholds[k]}"
		done
	done
done
