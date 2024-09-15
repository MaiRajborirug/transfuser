# /media/haoming/970EVO/carla$ ./CarlaUE4.sh --world-port=2000 -opengl
# conda activate tfuse-pr1
# /media/haoming/970EVO/Pharuj/git/transfuser/leaderboard/scripts$ ./eval2404.sh /media/haoming/970EVO/carla /media/haoming/970EVO/Pharuj/git/transfuser




# Initial setup
export CARLA_ROOT=${1:-/media/haoming/970EVO/carla}
export WORK_DIR=${2:-/media/haoming/970EVO/Pharuj/git/transfuser}

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export SCENARIOS=${WORK_DIR}/leaderboard/data/longest6/eval_scenarios.json
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=MAP # SENSORS , MAP
export ROUTES=/media/haoming/970EVO/Pharuj/git/transfuser/leaderboard/data/longest6/longest6_crashes2.xml

export TEAM_AGENT=${WORK_DIR}/team_code_transfuser/tfcbf_2404_noise.py # tf_2404_noise.py

export TEAM_CONFIG=/media/haoming/970EVO/Pharuj/transfuser_training/model_ckpt/models_2023/Transfuser_newweights/TransFuserAllTownsNoZeroNoSyncZGSeed1
export DEBUG_CHALLENGE=0
export RESUME=1
export DATAGEN=0
export PORT=2000
export TM_PORT=2500

# Loop for changing noise
# for NOISE in $(seq 0.0 0.1 1.0) # increment by 0.1 to 1.0
for NOISE in $(seq 0.0 0.1 0.0) # increment by 0.1 to 1.0
do
    export NOISE
    export SAVE_PATH="/media/haoming/970EVO/Pharuj/cdc_eval/240912_tfcbf_noise${NOISE}_rep${REPETITIONS}_5"
    export CHECKPOINT_ENDPOINT="${SAVE_PATH}.json"
    
    # Run the simulation with the current noise setting
    python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local_test.py \
    --scenarios=${SCENARIOS} \
    --routes=${ROUTES} \
    --repetitions=${REPETITIONS} \
    --track=${CHALLENGE_TRACK_CODENAME} \
    --checkpoint=${CHECKPOINT_ENDPOINT} \
    --agent=${TEAM_AGENT} \
    --agent-config=${TEAM_CONFIG} \
    --debug=${DEBUG_CHALLENGE} \
    --resume=${RESUME} \
    --port=${PORT} \
    --trafficManagerPort=${TM_PORT}
done

