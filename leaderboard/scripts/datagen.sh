export CARLA_ROOT=${1:-~/perception_based_control_alt/carla}
export WORK_DIR=${2:-/media/haoming/970EVO/Pharuj/git/transfuser}

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

# done TS = (7,10)
export Town=Town10HD
export Scenario=Scenario1

export SCENARIOS=${WORK_DIR}/leaderboard/data/training/scenarios/${Scenario}/${Town}_${Scenario}.json
export ROUTES=${WORK_DIR}/leaderboard/data/training/routes/${Scenario}/${Town}_${Scenario}.xml
export CHECKPOINT_ENDPOINT=/media/haoming/970EVO/Pharuj/transfuser_datagen/${Town}_${Scenario}.json  
export SAVE_PATH=/media/haoming/970EVO/Pharuj/transfuser_datagen/${Town}_${Scenario}


export TEAM_AGENT=${WORK_DIR}/team_code_autopilot/data_agent_copy.py
export CHALLENGE_TRACK_CODENAME=MAP
export REPETITIONS=1
export DEBUG_CHALLENGE=0
export RESUME=1
export DATAGEN=1

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--resume=${RESUME}
