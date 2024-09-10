export CARLA_ROOT=${1:-~/perception_based_control_alt/carla}
export WORK_DIR=${2:-/media/haoming/970EVO/Pharuj/git/transfuser}

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export SCENARIOS=${WORK_DIR}/leaderboard/data/longest6/eval_scenarios.json # config of 6 towns

export REPETITIONS=1
# export CHALLENGE_TRACK_CODENAME=SENSORS
export CHALLENGE_TRACK_CODENAME=MAP

# NOTE: checkpoint = eval result, agent = agent code, agent-config = model folder
export ROUTES=/media/haoming/970EVO/Pharuj/git/transfuser/leaderboard/data/longest6/longest6_crashes2.xml

# export CHECKPOINT_ENDPOINT=${WORK_DIR}/results/sc_training_dec/tfuseall_scratch_rl25e-5_26to35.json # train_eval
export SAVE_PATH=/media/haoming/970EVO/Pharuj/cdc_eval/240403_tfnoise0_1
export CHECKPOINT_ENDPOINT=${SAVE_PATH}.json # train_eval
export TEAM_AGENT=${WORK_DIR}/team_code_transfuser/submission_agent_adjust2.py # agent_240324.py submission_agent_adjust.py

# export TEAM_CONFIG=${WORK_DIR}/model_ckpt/models_2023/train_scratch_rl25e5  # transfuser_nosc
# export TEAM_CONFIG=/media/haoming/970EVO/Pharuj/transfuser_training/240118_gt1_ep35  # transfuser joint
export TEAM_CONFIG=/media/haoming/970EVO/Pharuj/transfuser_training/model_ckpt/models_2023/Transfuser_newweights/TransFuserAllTownsNoZeroNoSyncZGSeed1  # transfuser scratch

export DEBUG_CHALLENGE=0 # interfere with RGB cam
export RESUME=1
export DATAGEN=0
export PORT=2000
export TM_PORT=2500

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT} \

# work with scenario 29, 27, 26
