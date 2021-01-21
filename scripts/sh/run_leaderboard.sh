#!/bin/bash

# running on my local machine vs CMU cluster
export PREFIX=$8
source ~/anaconda3/etc/profile.d/conda.sh
conda activate lblbc

# Python env variables so the subdirectories can find each other
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/lib
#export CARLA_ROOT=$PREFIX/CARLA_0.9.10.1
export CARLA_ROOT=$PREFIX/CARLA_0.9.11
export LBC_ROOT=$PREFIX/2020_CARLA_challenge
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
#export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner
export DEBUG_CHALLENGE=0 # DO NOT MODIFY
export HAS_DISPLAY=0


# leaderboard and agent config
export PORT=$1
export ROUTES=$2
export LOGDIR=$3
export TM_PORT=$4
export AGENT=$5
export CONFIG=$6
export REPETITIONS=$7
export TEAM_AGENT=$LBC_ROOT/leaderboard/team_code/${AGENT}.py
export TEAM_CONFIG=$LBC_ROOT/leaderboard/config/${CONFIG}

# logging
if [ -d "$TEAM_CONFIG" ]; then
    CHECKPOINT_ENDPOINT="$LOGDIR/logs/$(basename $ROUTES .xml).txt"
else
    CHECKPOINT_ENDPOINT="$LOGDIR/logs/$(basename $ROUTES .xml).txt"
fi

python leaderboard/leaderboard/leaderboard_evaluator.py \
--track=SENSORS \
--scenarios=leaderboard/data/all_towns_traffic_scenarios_public.json  \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--routes=${ROUTES} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT} \
--debug=${DEBUG_CHALLENGE} \
--repetitions=${REPETITIONS}

echo "Done. See $CHECKPOINT_ENDPOINT for detailed results."

