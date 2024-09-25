#!/bin/bash

# Initial setup
export CARLA_ROOT=${1:-/media/haoming/970EVO/carla}
export WORK_DIR=${2:-/media/haoming/970EVO/pharuj/git/transfuser}

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export SCENARIOS=${WORK_DIR}/leaderboard/data/longest6/eval_scenarios.json
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=MAP # SENSORS, MAP
export ROUTES=/media/haoming/970EVO/pharuj/git/transfuser/leaderboard/data/longest6/longest6_crashes2.xml

# tf_2404_noise, tfcbf_2009_noise, tfcbf_2009_noise
export TEAM_AGENT=${WORK_DIR}/team_code_transfuser/tfsc_2404_noise.py

export TEAM_CONFIG=/media/haoming/970EVO/pharuj/transfuser_training/model_ckpt/models_2023/Transfuser_newweights/TransFuserAllTownsNoZeroNoSyncZGSeed1
export DEBUG_CHALLENGE=1 # for showing wp
export RESUME=1
export DATAGEN=0
export PORT=2000
export TM_PORT=2500

# Process command-line arguments for -safe-path
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -safe-path) SAFE_PATH="$2"; shift ;;
    esac
    shift
done

# Loop for changing noise (if required, you can modify NOISE loop logic)
for NOISE in $(seq 0.0 0.1 0.0) # increment by 0.1 up to 1.0
# for NOISE in [0.2 0. 1.0] # increment by 0.1 up to 1.0
do
    export NOISE
    
    # If SAFE_PATH is not provided, generate it automatically for each NOISE level
    if [ -z "$SAFE_PATH" ]; then
        # Get today's date
        DATE=$(date +%Y%m%d)
        
        # Extract TEAM_AGENT name without directory or extension
        TEAM_AGENT_NAME=$(basename ${TEAM_AGENT} .py)

        # Extract ROUTES name without directory or extension
        ROUTES_NAME=$(basename ${ROUTES} .xml)

        # Count how many times this specific configuration has been run today
        COUNT=1
        SAVE_FOLDER="/media/haoming/970EVO/pharuj/cdc_eval/"
        # SAVE_PATTERN="${SAVE_FOLDER}${DATE}-${TEAM_AGENT_NAME}-${NOISE}-${ROUTES_NAME}-"
        SAVE_PATTERN="${SAVE_FOLDER}${DATE}-test_wp-"
        while [ -d "${SAVE_PATTERN}${COUNT}" ]; do
            COUNT=$((COUNT + 1))
        done

        # Construct SAVE_PATH for the current NOISE value
        export SAVE_PATH="${SAVE_PATTERN}${COUNT}"
    else
        export SAVE_PATH="${SAFE_PATH}-${NOISE}"
    fi

    echo "SAVE_PATH set to: $SAVE_PATH"

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

# ./eval_2404.sh -safe-path "/media/haoming/970EVO/pharuj/custom_save_path"