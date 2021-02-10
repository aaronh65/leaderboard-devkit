#!/bin/bash

# running on my local machine vs CMU cluster
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

export CONFIG_PATH=$1

# Python env variables so the subdirectories can find each other
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/lib
export PYTHONPATH=$PYTHONPATH:$CARLA_EGG
export PYTHONPATH=$PYTHONPATH:$CARLA_API

export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/leaderboard
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/scenario_runner
export HAS_DISPLAY=0

python $PROJECT_ROOT/team_code/rl/$ALGO/trainer.py \
	--config_path=$CONFIG_PATH

echo "Done. See $SAVE_ROOT for detailed results."

