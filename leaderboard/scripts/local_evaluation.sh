export CARLA_ROOT=${1:-~/perception_based_control_alt/carla}
export WORK_DIR=${2:-~/git/transfuser}

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export SCENARIOS=${WORK_DIR}/leaderboard/data/longest6/eval_scenarios.json
export ROUTES=${WORK_DIR}/leaderboard/data/longest6_split/longest_weathers_12.xml
# 12
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=SENSORS

export CHECKPOINT_ENDPOINT=${WORK_DIR}/results/sc_training/weather12_nosc.json
export TEAM_CONFIG=${WORK_DIR}/model_ckpt/models_2022/transfuser_nosc

export TEAM_AGENT=${WORK_DIR}/team_code_transfuser/submission_agent.py
export DEBUG_CHALLENGE=0
export RESUME=1
export DATAGEN=0

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
