#!/usr/bin/env bash
set -euo pipefail

# Run this script from the repository root:
#   ./run_best_from_root.sh
# or:
#   bash run_best_from_root.sh
# Override training episodes:
#   ./run_best_from_root.sh --num_episodes 500
# Select models to run (comma-separated):
#   ./run_best_from_root.sh --models dqn,ddqn,qrdqn

NUM_EPISODES=300
MODELS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num_episodes)
      if [[ $# -lt 2 ]]; then
        echo "Error: --num_episodes requires a value" >&2
        exit 1
      fi
      NUM_EPISODES="$2"
      shift 2
      ;;
    --models)
      if [[ $# -lt 2 ]]; then
        echo "Error: --models requires a value" >&2
        exit 1
      fi
      MODELS="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--num_episodes N] [--models dqn,ddqn,dddqn,qrdqn,hdqn]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--num_episodes N] [--models dqn,ddqn,dddqn,qrdqn,hdqn]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$MODELS" ]]; then
  MODELS="dqn,ddqn,dddqn,qrdqn,hdqn"
fi

declare -A RUN_MODELS=(
  [dqn]=0
  [ddqn]=0
  [dddqn]=0
  [qrdqn]=0
  [hdqn]=0
)

IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
for raw_model in "${MODEL_LIST[@]}"; do
  model="${raw_model//[[:space:]]/}"
  model="${model,,}"
  case "$model" in
    dqn|ddqn|dddqn|qrdqn|hdqn)
      RUN_MODELS["$model"]=1
      ;;
    *)
      echo "Unknown model in --models: $model" >&2
      echo "Allowed models: dqn,ddqn,dddqn,qrdqn,hdqn" >&2
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

BASE_OUT="output"
mkdir -p "$BASE_OUT"

if [[ "${RUN_MODELS[dqn]}" -eq 1 ]]; then
  echo "Running best DQN hyperparameters..."
  uv run app/dqn.py \
    --num_episodes "$NUM_EPISODES" \
    --buffer_size 200000 \
    --batch_size 128 \
    --gamma 0.9556791110090774 \
    --lr 0.00014394430404112647 \
    --target_sync_every 250 \
    --learn_start 1000 \
    --learn_every 8 \
    --eps_start 1.0 \
    --eps_end 0.1488425606123552 \
    --eps_decay_steps 20000 \
    --max_steps_per_episode 5000 \
    --grad_clip 4.3076689475049506 \
    --num_eval_seeds 100 \
    --output_dir "$BASE_OUT/dqn"
fi

if [[ "${RUN_MODELS[ddqn]}" -eq 1 ]]; then
  echo "Running best Double DQN hyperparameters..."
  uv run app/double_dqn.py \
    --num_episodes "$NUM_EPISODES" \
    --buffer_size 20000 \
    --batch_size 256 \
    --gamma 0.9798426351600259 \
    --lr 0.0009768958892956907 \
    --target_sync_every 100 \
    --learn_start 1000 \
    --learn_every 2 \
    --eps_start 1.0 \
    --eps_end 0.17004595587593185 \
    --eps_decay_steps 50000 \
    --max_steps_per_episode 5000 \
    --grad_clip 6.668923925628828 \
    --num_eval_seeds 100 \
    --output_dir "$BASE_OUT/double-dqn"
fi

if [[ "${RUN_MODELS[dddqn]}" -eq 1 ]]; then
  echo "Running best Dueling Double DQN hyperparameters..."
  uv run app/dueling_double_dqn.py \
    --num_episodes "$NUM_EPISODES" \
    --buffer_size 200000 \
    --batch_size 64 \
    --gamma 0.9654707747743687 \
    --lr 0.0029222928006765354 \
    --target_sync_every 250 \
    --learn_start 2000 \
    --learn_every 8 \
    --eps_start 1.0 \
    --eps_end 0.19214154949975376 \
    --eps_decay_steps 10000 \
    --max_steps_per_episode 5000 \
    --grad_clip 1.1189121985227757 \
    --num_eval_seeds 100 \
    --output_dir "$BASE_OUT/dueling-double-dqn"
fi

if [[ "${RUN_MODELS[qrdqn]}" -eq 1 ]]; then
  echo "Running QR-DQN hyperparameters..."
  uv run app/qr_dqn.py \
    --num_episodes "$NUM_EPISODES" \
    --buffer_size 200000 \
    --batch_size 256 \
    --gamma 0.9922439589929455 \
    --lr 0.0001017167470236331 \
    --target_sync_every 1000 \
    --learn_start 5000 \
    --learn_every 2 \
    --eps_start 1.0 \
    --eps_end 0.16221083409453396 \
    --eps_decay_steps 20000 \
    --max_steps_per_episode 5000 \
    --grad_clip 15.05157300548717 \
    --num_quantiles 51 \
    --kappa 1.0 \
    --num_eval_seeds 100 \
    --output_dir "$BASE_OUT/qr-dqn"
fi

if [[ "${RUN_MODELS[hdqn]}" -eq 1 ]]; then
  echo "Running H-DQN hyperparameters..."
  uv run app/h_dqn.py \
    --num_episodes "$NUM_EPISODES" \
    --goal_tiles 32,64,128,256,512,1024,2048 \
    --option_duration 8 \
    --intrinsic_success_reward 1.0 \
    --intrinsic_step_penalty -0.01 \
    --ctrl_buffer_size 20000 \
    --ctrl_batch_size 256 \
    --ctrl_gamma 0.9974851723474096 \
    --ctrl_lr 0.00017931204091428885 \
    --ctrl_target_sync_every 250 \
    --ctrl_learn_start 5000 \
    --ctrl_learn_every 8 \
    --meta_buffer_size 20000 \
    --meta_batch_size 64 \
    --meta_gamma 0.9974851723474096 \
    --meta_lr 0.00017931204091428885 \
    --meta_target_sync_every 50 \
    --meta_learn_start 500 \
    --meta_learn_every 2 \
    --eps_start 1.0 \
    --eps_end 0.04539376238997539 \
    --eps_decay_steps 20000 \
    --meta_eps_start 1.0 \
    --meta_eps_end 0.04539376238997539 \
    --meta_eps_decay_steps 2000 \
    --max_steps_per_episode 5000 \
    --grad_clip 7.010189344276133 \
    --num_eval_seeds 100 \
    --output_dir "$BASE_OUT/h-dqn"
fi

echo "Selected models: $MODELS"
echo "All selected runs completed. Outputs are in: $BASE_OUT"