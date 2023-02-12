#!/usr/bin/env bash

function rename() {
	# Rename all files in a folder
	# $1: folder path
	# $2: starting number
	mkdir -p $1/renamed
	k=$2
	for folder in $1/episode_*; do
		if [ -d "$folder" ]; then
			echo "$folder -> $1/renamed/episode_$k"
			mv $folder $1/renamed/episode_$k
			k=$((k + 1))
		fi
	done
}
