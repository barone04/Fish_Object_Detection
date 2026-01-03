import torch
import time
import json
import os
from thop import profile
from models.faster_rcnn import fasterrcnn_resnet50_fpn, fasterrcnn_resnet18_fpn

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Kiểm tra kỹ đường dẫn file của bạn
BASELINE_PATH = "output/pipeline2_fpn/step1_dense_det/model_best.pth"
PRUNED_PATH = "output/pipeline2_fpn/step3_final_result/model_best.pth"
PRUNED_CONFIG = "output/pipeline2_fpn/step2_pruned_det/model_lean.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def measure_model(model_name, weights, config=None):
    print(f"\nMeasuring: {model_name}...")

    # --- 1. XỬ LÝ CONFIG (Fix lỗi KeyError) ---
    backbone_rates = None
    fpn_rates = None

    if config:
        if os.path.exists(config):
            with open(config, 'r') as f:
                config_data = json.load(f)

            # Logic tách Backbone và FPN rates
            if isinstance(config_data, dict):
                backbone_rates = config_data.get('backbone')
                fpn_rates = config_data.get('fpn')
                print(f"   Config loaded: Backbone rates={len(backbone_rates)}, FPN rates={len(fpn_rates)}")
            elif isinstance(config_data, list):
                backbone_rates = config_data
                fpn_rates = None
                print(f"   Config loaded (Old format): Backbone rates={len(backbone_rates)}")
        else:
            print(f"   Warning: Config file not found at {config}")

    # --- 2. KHỞI TẠO MODEL ---
    model = fasterrcnn_resnet18_fpn(
        num_classes=2,
        compress_rate=backbone_rates,
        fpn_compress_rate=fpn_rates
    )

    # --- 3. LOAD WEIGHTS ---
    if weights:
        if os.path.exists(weights):
            print(f"   Loading weights from {weights}")
            try:
                # weights_only=False để fix warning trên các bản PyTorch mới
                try:
                    ckpt = torch.load(weights, map_location='cpu', weights_only=False)
                except TypeError:
                    ckpt = torch.load(weights, map_location='cpu')

                if isinstance(ckpt, dict) and 'model' in ckpt:
                    ckpt = ckpt['model']

                model.load_state_dict(ckpt, strict=False)
            except Exception as e:
                print(f"   Error loading weights: {e}")
        else:
            print(f"   Warning: Weights file not found at {weights}. Using Random Init.")

    model.to(DEVICE)
    # Set eval lần 1
    model.eval()

    # --- 4. ĐO PARAMS & FLOPs (Fix lỗi input dimension) ---
    # FasterRCNN yêu cầu list các 3D tensor [C, H, W]
    dummy_tensor_3d = torch.randn(3, 800, 800).to(DEVICE)

    # Đo Params
    params = sum(p.numel() for p in model.parameters())
    flops = 0

    try:
        # Wrapper class để thop hiểu được input dạng list của FasterRCNN
        class Wrapper(torch.nn.Module):
            def __init__(self, m): super().__init__(); self.m = m

            def forward(self, x):
                return self.m([x])  # Tự động bọc input vào List

        wrapped_model = Wrapper(model)
        # thop sẽ truyền dummy_tensor_3d vào hàm forward của wrapper
        flops, _ = profile(wrapped_model, inputs=(dummy_tensor_3d,), verbose=False)
    except Exception as e:
        print(f"   Warning: Could not measure FLOPs ({e}). Skipping.")

    # --- 5. ĐO FPS (Fix lỗi targets is None) ---
    try:
        # QUAN TRỌNG: Ép kiểu về Eval Mode một lần nữa để chắc chắn
        # (Vì thop hoặc Wrapper có thể đã vô tình reset trạng thái)
        model.eval()

        # Input chuẩn bị cho loop đo FPS
        dummy_input_list = [dummy_tensor_3d]

        # Warmup (Làm nóng GPU)
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input_list)

        # Run Benchmark
        iters = 50
        if DEVICE == "cuda": torch.cuda.synchronize()
        t_start = time.time()

        with torch.no_grad():
            for _ in range(iters):
                _ = model(dummy_input_list)

        if DEVICE == "cuda": torch.cuda.synchronize()
        t_end = time.time()

        fps = iters / (t_end - t_start)
    except Exception as e:
        print(f"   Error measuring FPS: {e}")
        fps = 0

    print(f"   Done. Params: {params / 1e6:.2f}M | FLOPs: {flops / 1e9:.2f}G | FPS: {fps:.2f}")

    return params, flops, fps


def main():
    try:
        print("=" * 65)
        print("BENCHMARKING MODEL PERFORMANCE")
        print("=" * 65)

        # 1. Measure Baseline
        p_base, f_base, fps_base = measure_model("Baseline (Dense)", BASELINE_PATH, None)

        # 2. Measure Pruned
        p_pruned, f_pruned, fps_pruned = measure_model("Pruned (Lean)", PRUNED_PATH, PRUNED_CONFIG)

        print("\n" + "=" * 65)
        print(f"{'Metric':<20} | {'Baseline':<15} | {'Pruned':<15} | {'Improvement':<10}")
        print("-" * 65)

        # Params Calculation
        impr_p = (1 - p_pruned / p_base) * 100 if p_base > 0 else 0
        print(f"{'Parameters (M)':<20} | {p_base / 1e6:<15.2f} | {p_pruned / 1e6:<15.2f} | -{impr_p:.2f}%")

        # FLOPs Calculation
        impr_f = (1 - f_pruned / f_base) * 100 if f_base > 0 else 0
        print(f"{'FLOPs (G)':<20} | {f_base / 1e9:<15.2f} | {f_pruned / 1e9:<15.2f} | -{impr_f:.2f}%")

        # FPS Calculation
        impr_fps = (fps_pruned / fps_base - 1) * 100 if fps_base > 0 else 0
        print(f"{'FPS (800x800)':<20} | {fps_base:<15.2f} | {fps_pruned:<15.2f} | +{impr_fps:.2f}%")
        print("=" * 65)

    except Exception as e:
        print(f"\nCRITICAL ERROR IN MAIN: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()