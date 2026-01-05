#!/bin/bash

set -e
set -o pipefail

# experiment ID
RUN_ID=$(date +"%Y%m%d_%H%M%S")

# GPU selection
export CUDA_VISIBLE_DEVICES=0

# dir
LOG_DIR="logs/train/${RUN_ID}"
CHECKPOINT_DIR="checkpoints/${RUN_ID}"
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

# parameters
ARCH="resnet18"
EPOCHS=100
LR=0.0001
SEED=1234
PATIENCE=20
DEVICE="cuda"

echo "Experiment ID: $RUN_ID"
echo "Logs: $LOG_DIR"
echo "Checkpoints: $CHECKPOINT_DIR"

# H0 - main
echo "[1/4] Training H0 (Main)"
python src/train/train.py \
  --arch $ARCH \
  --epochs $EPOCHS \
  --lr $LR \
  --seed $SEED \
  --patience $PATIENCE \
  --device $DEVICE \
  --output "${CHECKPOINT_DIR}/h0_main" \
  2>&1 | tee "${LOG_DIR}/h0_main.log"

# H1 - straight
echo "[2/4] Training H1 (Straight)"
python src/train/train.py \
  --arch $ARCH \
  --epochs $EPOCHS \
  --lr $LR \
  --seed $SEED \
  --patience $PATIENCE \
  --device $DEVICE \
  --parent-class 1 \
  --output "${CHECKPOINT_DIR}/h1_straight" \
  2>&1 | tee "${LOG_DIR}/h1_straight.log"

# H2 - wavy
echo "[3/4] Training H2 (Wavy)"
python src/train/train.py \
  --arch $ARCH \
  --epochs $EPOCHS \
  --lr $LR \
  --seed $SEED \
  --patience $PATIENCE \
  --device $DEVICE \
  --parent-class 2 \
  --output "${CHECKPOINT_DIR}/h2_wavy" \
  2>&1 | tee "${LOG_DIR}/h2_wavy.log"

# H3 - curly
echo "[4/4] Training H3 (Curly)"
python src/train/train.py \
  --arch $ARCH \
  --epochs $EPOCHS \
  --lr $LR \
  --seed $SEED \
  --patience $PATIENCE \
  --device $DEVICE \
  --parent-class 3 \
  --output "${CHECKPOINT_DIR}/h3_curly" \
  2>&1 | tee "${LOG_DIR}/h3_curly.log"

echo "All trainings finished. Logs: $LOG_DIR, Checkpoints: $CHECKPOINT_DIR"
