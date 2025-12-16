#!/bin/bash
set -e

# --- CẤU HÌNH ---
DATA_ROOT="./dataset/fish_detection"
OUTPUT_DIR="./output/baseline_dense"
BATCH_SIZE=32 # Giữ nguyên như lúc train prune
WORKERS=4     # Giữ nguyên để tránh lỗi ancdata

echo "======================================================="
echo "STARTING BASELINE TRAINING (DENSE MODEL)"
echo "Device: H100 | Batch: $BATCH_SIZE | Epochs: 50"
echo "======================================================="

# Lưu ý: KHÔNG truyền --compress-rate và --weights-backbone
# Code sẽ tự hiểu là Dense Model + ImageNet Init
python train_det.py \
    --data-path $DATA_ROOT \
    --epochs 30 \
    --batch-size $BATCH_SIZE \
    --workers $WORKERS \
    --lr 0.02 \
    --output-dir $OUTPUT_DIR

echo "======================================================="
echo "BASELINE COMPLETED!"
echo "Model: $OUTPUT_DIR/model_best.pth"
echo "======================================================="