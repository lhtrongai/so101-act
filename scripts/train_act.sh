#!/usr/bin/env bash
# Train ACT with LeRobot on a GPU pod, in the background.
# Usage:
#   bash scripts/train_act.sh <dataset_repo_id> <policy_repo_id> [steps] [extra lerobot-train flags...]
# Examples:
#   bash scripts/train_act.sh tronghle/so101_pick_red_trimmed tronghle/act_so101_pick_red
#   bash scripts/train_act.sh tronghle/so101_pick_red_trimmed tronghle/act_so101_pick_red_aug 100000 \
#       --dataset.image_transforms.enable=true
# Benchmark first with steps=200 and read data_s against updt_s in the log.
set -euo pipefail

DATASET=$1
POLICY_REPO=$2
STEPS=${3:-100000}
shift $(( $# >= 3 ? 3 : $# ))
NAME=$(basename "$POLICY_REPO")
OUT=/root/outputs/$NAME
LOG=/root/$NAME.log

nohup lerobot-train \
  --dataset.repo_id="$DATASET" \
  --policy.type=act --policy.device=cuda \
  --policy.push_to_hub=true --policy.repo_id="$POLICY_REPO" --policy.private=true \
  --output_dir="$OUT" --job_name="$NAME" \
  --steps="$STEPS" --batch_size=8 --num_workers=16 \
  --log_freq=500 --save_freq=20000 \
  --trackio.enable=true --trackio.space_id=tronghle/so101-act-runs --trackio.private=true \
  "$@" > "$LOG" 2>&1 &

echo "Training in the background. Progress: tail -n 2 $LOG"
echo "When done: bash scripts/upload_checkpoints.sh $POLICY_REPO"
