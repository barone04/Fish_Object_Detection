#!/bin/bash
set -e

# --- CẤU HÌNH ---
DATA_ROOT="./NewDeepfish/NewDeepfish"
OUTPUT_ROOT="./output/pipeline2"
BATCH_SIZE=8
WORKERS=16

echo "======================================================="
echo "STARTING PIPELINE 2: END-TO-END DETECTION PRUNING"
echo "Batch Size: $BATCH_SIZE"
echo "======================================================="

# ---------------------------------------------------------
# BƯỚC 1: Train Dense Detection Model
# Mục tiêu: Có một mô hình Detection chuẩn (mAP cao nhất)
# ---------------------------------------------------------
echo "[Step 1/3] Skipping Dense Training (Using existing weights)..."
#echo "[Step 1/3] Training Dense Faster R-CNN..."
#python train_det.py \
#    --data-path $DATA_ROOT \
#    --epochs 60 \
#    --batch-size $BATCH_SIZE \
#    --workers $WORKERS \
#    --lr 0.01 \
#    --output-dir $OUTPUT_ROOT/step1_dense_det

# ---------------------------------------------------------
# BƯỚC 2: Pruning Loop (Trái tim của dự án)
# Mục tiêu: Vừa cắt vừa train lại để giữ mAP detection
# --checkpoint $OUTPUT_ROOT/step1_dense_det/model_best.pth \
# ---------------------------------------------------------
echo "[Step 2/3] Iterative Pruning (SongHan + Filter)..."
# Lưu ý: Script này sẽ tự động gọi Surgery ở cuối để tạo model_lean.pth
python prune_det.py \
    --data-path $DATA_ROOT \
    --checkpoint ./output/step_1_dense_model/model_best.pth \
    --target-sparsity 0.5 \
    --prune-iters 8 \
    --finetune-epochs 10 \
    --batch-size $BATCH_SIZE \
    --output-dir $OUTPUT_ROOT/step2_pruned_det

# ---------------------------------------------------------
# BƯỚC 3: Final Finetuning
# ---------------------------------------------------------
echo "[Step 3/3] Final Finetuning..."
python train_det.py \
    --data-path $DATA_ROOT \
    --weights-backbone $OUTPUT_ROOT/step2_pruned_det/backbone_lean.pth \
    --compress-rate $OUTPUT_ROOT/step2_pruned_det/backbone_lean.json \
    --epochs 50 \
    --batch-size $BATCH_SIZE \
    --workers $WORKERS \
    --lr 0.02 \
    --lr-steps 100 130 \
    --output-dir $OUTPUT_ROOT/step3_final_result

echo "======================================================="
echo "PIPELINE 2 COMPLETED SUCCESSFULLY!"
echo "Final Model: $OUTPUT_ROOT/step3_final_result/model_best.pth"
echo "======================================================="