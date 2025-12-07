#!/bin/bash
set -e

# --- CẤU HÌNH ---
DATA_ROOT="./dataset/fish_detection"
OUTPUT_ROOT="./output/pipeline2"
BATCH_SIZE=32 # Faster R-CNN tốn VRAM, 32 là con số đẹp cho H100 (80GB)
WORKERS=16

echo "======================================================="
echo "STARTING PIPELINE 2: END-TO-END DETECTION PRUNING"
echo "Device: H100 | Batch Size: $BATCH_SIZE"
echo "======================================================="

# ---------------------------------------------------------
# BƯỚC 1: Train Dense Detection Model
# Mục tiêu: Có một mô hình Detection chuẩn (mAP cao nhất)
# ---------------------------------------------------------
echo "[Step 1/3] Training Dense Faster R-CNN..."
python train_det.py \
    --data-path $DATA_ROOT \
    --epochs 50 \
    --batch-size $BATCH_SIZE \
    --workers $WORKERS \
    --lr 0.02 \
    --output-dir $OUTPUT_ROOT/step1_dense_det

# ---------------------------------------------------------
# BƯỚC 2: Pruning Loop (Trái tim của dự án)
# Mục tiêu: Vừa cắt vừa train lại để giữ mAP detection
# ---------------------------------------------------------
echo "[Step 2/3] Iterative Pruning (SongHan + Filter)..."
# Lưu ý: Script này sẽ tự động gọi Surgery ở cuối để tạo model_lean.pth
python prune_det.py \
    --data-path $DATA_ROOT \
    --checkpoint $OUTPUT_ROOT/step1_dense_det/model_best.pth \
    --target-sparsity 0.6 \
    --prune-iters 10 \
    --finetune-epochs 2 \
    --batch-size $BATCH_SIZE \
    --output-dir $OUTPUT_ROOT/step2_pruned_det

# ---------------------------------------------------------
# BƯỚC 3: Final Finetuning
# Mục tiêu: Ổn định Batch Norm và trọng số sau khi phẫu thuật
# ---------------------------------------------------------
echo "[Step 3/3] Final Finetuning..."
python train_det.py \
    --data-path $DATA_ROOT \
    --weights $OUTPUT_ROOT/step2_pruned_det/model_lean.pth \
    --compress-rate $OUTPUT_ROOT/step2_pruned_det/config.json \
    --epochs 20 \
    --batch-size $BATCH_SIZE \
    --workers $WORKERS \
    --lr 0.002 \
    --output-dir $OUTPUT_ROOT/step3_final_result

echo "======================================================="
echo "PIPELINE 2 COMPLETED SUCCESSFULLY!"
echo "Final Model: $OUTPUT_ROOT/step3_final_result/model_best.pth"
echo "======================================================="