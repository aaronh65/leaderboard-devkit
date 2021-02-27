#!/usr/bin/env bash

CARLA_ROOT=/home/aaron/workspace/carla/CARLA_0.9.11
CARLA_API=$CARLA_ROOT/PythonAPI/carla
CARLA_EGG=$CARLA_API/dist/carla-0.9.11-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:$CARLA_EGG
export PYTHONPATH=$PYTHONPATH:$CARLA_API

PROJECT_ROOT=/home/aaron/workspace/carla/leaderboard-devkit
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/team_code
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/leaderboard
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/scenario_runner

