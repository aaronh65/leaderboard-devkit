#!/bin/bash

# running on my local machine vs CMU cluster
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# Python env variables so the subdirectories can find each other
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/lib
export PYTHONPATH=$PYTHONPATH:$CARLA_EGG
export PYTHONPATH=$PYTHONPATH:$CARLA_API

export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/leaderboard
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/scenario_runner

export DEBUG_CHALLENGE=0 # DO NOT MODIFY

# leaderboard and agent config
TEAM_CONFIG=$SAVE_ROOT/config.yml
TEAM_AGENT=$PROJECT_ROOT/team_code/$AGENT.py
ROUTE_PATH=$PROJECT_ROOT/leaderboard/data/routes_$SPLIT/$ROUTE_NAME.xml
SCENARIOS=$PROJECT_ROOT/leaderboard/data/all_towns_traffic_scenarios_public.json

CHECKPOINT_ENDPOINT="$SAVE_ROOT/logs/${ROUTE_NAME}.txt"

python ${PROJECT_ROOT}/leaderboard/leaderboard/leaderboard_evaluator.py \
--track=SENSORS \
--scenarios=${SCENARIOS} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--routes=${ROUTE_PATH} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--port=${WORLD_PORT} \
--trafficManagerPort=${TM_PORT} \
--debug=${DEBUG_CHALLENGE} \
--repetitions=${REPETITIONS} \
--privileged=${PRIVILEGED}

echo "Done. See $CHECKPOINT_ENDPOINT for detailed results."

