# Runing this pipeline: bash scripts/pipeline1_transfer.sh

#!/bin/bash
set -e  # Dừng ngay nếu có lỗi

# --- CẤU HÌNH ---
DATA_ROOT="./dataset/fish_detection" # Đường dẫn tới thư mục dataset gốc
OUTPUT_ROOT="./output/pipeline1"
GPUS=1
# H100 rất mạnh, batch size lớn giúp Batch Norm ổn định và train nhanh
BATCH_SIZE_CLS=256
BATCH_SIZE_DET=32
WORKERS=16 # Tận dụng 20 vCPU

echo "======================================================="
echo "STARTING PIPELINE 1: TRANSFER PRUNING"
echo "Device: H100 | Batch Cls: $BATCH_SIZE_CLS | Batch Det: $BATCH_SIZE_DET"
echo "======================================================="

# ---------------------------------------------------------
# BƯỚC 1: Train Dense Backbone (Classification Task)
# Mục tiêu: Backbone học đặc trưng loài cá
# ---------------------------------------------------------
echo "[Step 1/3] Training Dense Backbone..."
python train_cls.py \
    --data-path $DATA_ROOT \
    --model resnet50 \
    --epochs 20 \
    --batch-size $BATCH_SIZE_CLS \
    --workers $WORKERS \
    --lr 0.1 \
    --output-dir $OUTPUT_ROOT/step1_dense_cls

# ---------------------------------------------------------
# BƯỚC 2: Prune Backbone (SongHan + Filter Pruning)
# Mục tiêu: Tạo backbone nhỏ gọn (Lean Model)
# ---------------------------------------------------------
echo "[Step 2/3] Pruning Backbone..."
python prune_cls.py \
    --data-path $DATA_ROOT \
    --checkpoint $OUTPUT_ROOT/step1_dense_cls/model_best.pth \
    --target-sparsity 0.6 \
    --prune-iters 5 \
    --output-dir $OUTPUT_ROOT/step2_pruned_cls
# Output mong đợi: backbone_lean.pth và config.json

# ---------------------------------------------------------
# BƯỚC 3: Transfer Learning sang Detection
# Mục tiêu: Dùng backbone nén để train Faster R-CNN
# ---------------------------------------------------------
echo "[Step 3/3] Training Faster R-CNN with Pruned Backbone..."
python train_det.py \
    --data-path $DATA_ROOT \
    --weights-backbone $OUTPUT_ROOT/step2_pruned_cls/backbone_lean.pth \
    --compress-rate $OUTPUT_ROOT/step2_pruned_cls/config.json \
    --epochs 50 \
    --batch-size $BATCH_SIZE_DET \
    --workers $WORKERS \
    --lr 0.02 \
    --output-dir $OUTPUT_ROOT/step3_final_det

echo "======================================================="
echo "PIPELINE 1 COMPLETED SUCCESSFULLY!"
echo "Final Model: $OUTPUT_ROOT/step3_final_det/model_best.pth"
echo "======================================================="