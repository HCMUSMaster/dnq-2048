#!/usr/bin/env bash
set -euo pipefail

# Run this script from the repository root:
#   ./run_best_from_root.sh
# or:
#   bash run_best_from_root.sh
# Override training episodes:
#   ./run_best_from_root.sh --num_episodes 500

NUM_EPISODES=300

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
    -h|--help)
      echo "Usage: $0 [--num_episodes N]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--num_episodes N]" >&2
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

BASE_OUT="output"
mkdir -p "$BASE_OUT"

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

echo "All runs completed. Outputs are in: $BASE_OUT"