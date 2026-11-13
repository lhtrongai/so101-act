#!/usr/bin/env bash
# Upload every checkpoint (weights only) and the training log to a private Hub repo
# "<policy_repo_id>_checkpoints". Run this before stopping the pod: its disk is wiped on stop.
# Usage: bash scripts/upload_checkpoints.sh <policy_repo_id>
set -euo pipefail

POLICY_REPO=$1
NAME=$(basename "$POLICY_REPO")
OUT=/root/outputs/$NAME

cp "/root/$NAME.log" "$OUT/train.log"
hf upload "${POLICY_REPO}_checkpoints" "$OUT" --repo-type model --private \
  --include "checkpoints/*/pretrained_model/*" --include "train.log"
echo "Uploaded. Stop or terminate the pod now."
