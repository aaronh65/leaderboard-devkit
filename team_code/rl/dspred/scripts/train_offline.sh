#!/bin/bash

# Generic run script that requires a command, a config location, and a conda env

# running on my local machine vs CMU cluster
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dspred

export DATASET_DIR=$1
export PROJECT_ROOT=/home/aaron/workspace/carla/leaderboard-devkit

# Python env variables so the subdirectories can find each other
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/lib
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/leaderboard
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/scenario_runner
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/team_code
#export HAS_DISPLAY=0

python $PROJECT_ROOT/team_code/rl/dspred/map_model.py \
	--dataset_dir=$DATASET_DIR
