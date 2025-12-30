import torch
import os
import json
import time
from engines import trainer_det, utils as engine_utils
from data.fish_det_dataset import FishDetectionDataset, collate_fn
from data import presets

# --- [QUAN TRỌNG] Import đúng hàm builder cho ResNet18 ---
from models.faster_rcnn import fasterrcnn_resnet18_fpn

# ======================================================================================
#                                   PHẦN CẤU HÌNH (SỬA TẠI ĐÂY)
# ======================================================================================

# 1. Đường dẫn Dataset
DATA_PATH = "./NewDeepfish/NewDeepfish"

# 2. Đường dẫn Model Step 1 (Dense - Chưa cắt)
DENSE_MODEL_PATH = "model_best.pth"

# 3. Đường dẫn Model Step 3 (Pruned - Đã cắt và finetune)
PRUNED_MODEL_PATH = "./output/pipeline2/step3_final_result/model_best.pth"

# 4. Đường dẫn file Config JSON (Sinh ra ở Step 2)
# Lưu ý: File này chứa tỷ lệ cắt để khởi tạo khung mạng cho model pruned
PRUNED_CONFIG_PATH = "./output/pipeline2/step2_pruned_det/backbone_lean.json"

# 5. Các cài đặt khác
NUM_CLASSES = 2  # Background + Fish
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')# hoặc 'cpu'
BATCH_SIZE = 8  # Batch size khi test (có thể tăng lên 16 nếu VRAM đủ)
NUM_WORKERS = 4


# ======================================================================================

def load_model_resnet18(checkpoint_path, config_json_path=None, num_classes=2, device='cuda'):
    """
    Hàm load model ResNet18 thông minh:
    - Nếu config_json_path=None -> Load Dense Model.
    - Nếu có config_json_path   -> Load Pruned Model (Lean).
    """
    print(f"\n[{'PRUNED' if config_json_path else 'DENSE'}] Loading from: {checkpoint_path}")

    # 1. Load Config cắt tỉa (nếu có)
    compress_rate_list = None
    if config_json_path:
        if os.path.exists(config_json_path):
            print(f"   -> Found pruning config: {config_json_path}")
            with open(config_json_path, 'r') as f:
                compress_rate_list = json.load(f)
        else:
            print(f"   -> WARNING: Config file not found at {config_json_path}")

    # 2. Khởi tạo kiến trúc mạng (Backbone)
    # Lưu ý: Gọi đúng hàm fasterrcnn_resnet18_fpn
    model = fasterrcnn_resnet18_fpn(
        num_classes=num_classes,
        compress_rate=compress_rate_list,
        weights_backbone=None
    )

    # 3. Load Trọng số (Weights)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Xử lý key bọc ngoài (thường là 'model' hoặc 'state_dict')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Load vào model
        try:
            model.load_state_dict(state_dict, strict=False)
            print("   -> Weights loaded successfully.")
        except Exception as e:
            print(f"   -> ERROR loading weights: {e}")
    else:
        print(f"   -> ERROR: Checkpoint file not found at {checkpoint_path}")

    model.to(device)
    return model


def evaluate_performance(model, data_loader, device):
    """
    Chạy đánh giá mAP trên tập validation
    """
    print("   -> Starting Evaluation...")
    t0 = time.time()

    # Sử dụng hàm evaluate có sẵn trong trainer_det của bạn
    coco_evaluator = trainer_det.evaluate(model, data_loader, device=device)

    t1 = time.time()
    eval_time = t1 - t0

    # Trích xuất chỉ số mAP
    # stats[0] = mAP @ 0.50:0.95
    # stats[1] = mAP @ 0.50
    map_standard = 0.0
    map_50 = 0.0

    if coco_evaluator is not None and hasattr(coco_evaluator, 'coco_eval'):
        stats = coco_evaluator.coco_eval['bbox'].stats
        map_standard = stats[0]
        map_50 = stats[1]

    print(f"   -> Done in {eval_time:.1f}s. mAP={map_standard:.4f}, mAP@50={map_50:.4f}")
    return map_standard, map_50, eval_time


def main():
    print("=" * 60)
    print("       RESNET18: DENSE vs PRUNED COMPARISON")
    print("=" * 60)

    # 1. Chuẩn bị Dữ liệu (Chỉ tập Validation)
    print("1. Preparing Data...")
    dataset_test = FishDetectionDataset(DATA_PATH, split='val', transforms=presets.DetectionPresetEval())
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=BATCH_SIZE, sampler=test_sampler,
        num_workers=NUM_WORKERS, collate_fn=collate_fn
    )
    print(f"   -> Found {len(dataset_test)} validation images.")

    # 2. Đánh giá Dense Model
    print("\n2. Evaluating DENSE Model (Step 1)...")
    model_dense = load_model_resnet18(DENSE_MODEL_PATH, config_json_path=None, num_classes=NUM_CLASSES, device=DEVICE)
    map_dense, map50_dense, time_dense = evaluate_performance(model_dense, data_loader_test, DEVICE)

    # Xóa model dense để giải phóng VRAM cho model pruned
    del model_dense
    torch.cuda.empty_cache()

    # 3. Đánh giá Pruned Model
    print("\n3. Evaluating PRUNED Model (Step 3)...")
    # Lưu ý: Bắt buộc phải truyền PRUNED_CONFIG_PATH để model khởi tạo đúng cấu trúc nhỏ gọn
    model_pruned = load_model_resnet18(PRUNED_MODEL_PATH, config_json_path=PRUNED_CONFIG_PATH, num_classes=NUM_CLASSES,
                                       device=DEVICE)
    map_pruned, map50_pruned, time_pruned = evaluate_performance(model_pruned, data_loader_test, DEVICE)

    # 4. In bảng so sánh
    print("\n" + "=" * 80)
    print(f"{'METRIC':<20} | {'DENSE MODEL (Step 1)':<22} | {'PRUNED MODEL (Step 3)':<22} | {'CHANGE'}")
    print("-" * 80)

    # So sánh mAP (0.5:0.95)
    diff_map = map_pruned - map_dense
    print(f"{'mAP (0.5:0.95)':<20} | {map_dense:.4f}{' ' * 16} | {map_pruned:.4f}{' ' * 16} | {diff_map:+.4f}")

    # So sánh mAP@50
    diff_map50 = map50_pruned - map50_dense
    print(f"{'mAP @ 0.50':<20} | {map50_dense:.4f}{' ' * 16} | {map50_pruned:.4f}{' ' * 16} | {diff_map50:+.4f}")

    # So sánh thời gian (tham khảo)
    diff_time = time_pruned - time_dense
    print(f"{'Eval Time (s)':<20} | {time_dense:.1f}s{' ' * 15} | {time_pruned:.1f}s{' ' * 15} | {diff_time:+.1f}s")

    print("-" * 80)

    # Kết luận nhanh
    if diff_map >= -0.02:  # Giảm ít hơn 2%
        print(">> KẾT LUẬN: Pruning thành công! mAP được giữ vững.")
    elif diff_map >= -0.05:
        print(">> KẾT LUẬN: Chấp nhận được. mAP giảm nhẹ.")
    else:
        print(">> KẾT LUẬN: Cảnh báo! mAP giảm nhiều, cần train Step 3 lâu hơn hoặc giảm tỷ lệ cắt.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()