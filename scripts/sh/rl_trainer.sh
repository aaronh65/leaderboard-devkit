#!/bin/bash

# running on my local machine vs CMU cluster
export NAME=aaron
source /home/$NAME/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

export CONFIG_PATH=$1

# Python env variables so the subdirectories can find each other
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$NAME/anaconda3/lib
export CARLA_ROOT=/home/$NAME/workspace/carla/CARLA_0.9.10.1
#export CARLA_ROOT=/home/$NAME/workspace/carla/CARLA_0.9.11
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
#export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg

export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/leaderboard
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/scenario_runner
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/team_code

python $PROJECT_ROOT/leaderboard/team_code/rl/trainer.py \
	--config_path=$CONFIG_PATH

echo "Done. See $BASE_SAVE_PATH for detailed results."

