#!/bin/bash

# Generic run script that requires a command, a config location, and a conda env

# running on my local machine vs CMU cluster
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

export SCRIPT=$1
export CONFIG_PATH=$2

# Python env variables so the subdirectories can find each other
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/lib
export PYTHONPATH=$PYTHONPATH:$CARLA_EGG
export PYTHONPATH=$PYTHONPATH:$CARLA_API

export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/leaderboard
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/scenario_runner
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/team_code
#export HAS_DISPLAY=0

python $SCRIPT \
	--config_path=$CONFIG_PATH

echo "Done running python $SCRIPT --config_path=$CONFIG_PATH"

