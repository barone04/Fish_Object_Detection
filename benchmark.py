import torch
import time
import json
import os
from thop import profile
from models.faster_rcnn import fasterrcnn_resnet50_fpn

# Path
BASELINE_PATH = "model_29.pth"
PRUNED_PATH = "output/step_1_dense_model/model_best.pth"
PRUNED_CONFIG = "backbone_lean.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def measure_model(model_name, weights, config=None):
    print(f"\n>>> Measuring: {model_name}...")

    # 1. Load Model
    print("   [Step 1] Loading Model & Config...")
    cpr = None
    if config:
        if os.path.exists(config):
            with open(config, 'r') as f:
                cpr = json.load(f)
        else:
            print(f"Warning: Config file not found at {config}")

    model = fasterrcnn_resnet50_fpn(num_classes=14, compress_rate=cpr)

    if weights:
        if os.path.exists(weights):
            print(f"   Loading weights from {weights}")
            # Fix lỗi bảo mật pickle bằng weights_only=False
            try:
                ckpt = torch.load(weights, map_location='cpu', weights_only=False)
            except TypeError:
                ckpt = torch.load(weights, map_location='cpu')

            if 'model' in ckpt: ckpt = ckpt['model']
            model.load_state_dict(ckpt)
        else:
            print(f"   Warning: Weights file not found at {weights}. Using Random Init.")

    model.to(DEVICE)

    # QUAN TRỌNG: Phải chuyển sang EVAL mode ngay lập tức
    model.eval()

    # 2. Measure Params & FLOPs
    print("   [Step 2] Measuring Params & FLOPs...")

    # Chuẩn bị Dummy Input chuẩn cho Faster R-CNN (List of 3D Tensors)
    dummy_tensor = torch.randn(3, 800, 800).to(DEVICE)
    dummy_input_list = [dummy_tensor]

    # Đo Params
    params = sum(p.numel() for p in model.parameters())
    flops = 0

    # Đo FLOPs (dùng thop)
    try:
        # Wrapper để thop hiểu list input
        class Wrapper(torch.nn.Module):
            def __init__(self, m): super().__init__(); self.m = m

            def forward(self, x): return self.m([x])

        wrapped_model = Wrapper(model)
        flops, _ = profile(wrapped_model, inputs=(dummy_tensor,), verbose=False)
    except Exception as e:
        print(f"   Warning: Could not measure FLOPs ({e}). Skipping.")

    # 3. Measure FPS
    print("   [Step 3] Measuring FPS (Warming up)...")
    model.eval()

    try:
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_input_list)

        # Run
        print("   Benchmarking...")
        iters = 50  # Đo 50 lần lấy trung bình
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

    print(f"   -> Done. Params: {params / 1e6:.2f}M | FLOPs: {flops / 1e9:.2f}G | FPS: {fps:.2f}")

    # QUAN TRỌNG: Dòng này bắt buộc phải có và thụt lề đúng
    return params, flops, fps


def main():
    print("=======================================================")
    print("STARTING BENCHMARK COMPARISON")
    print("=======================================================")

    try:
        # 1. Measure Baseline
        p_base, f_base, fps_base = measure_model("Baseline (Dense)", BASELINE_PATH, None)

        # 2. Measure Pruned
        p_pruned, f_pruned, fps_pruned = measure_model("Pruned (Lean)", PRUNED_PATH, PRUNED_CONFIG)

        print("\n" + "=" * 65)
        print(f"{'Metric':<20} | {'Baseline':<15} | {'Pruned':<15} | {'Improvement':<10}")
        print("-" * 65)

        # Params
        impr_p = (1 - p_pruned / p_base) * 100 if p_base > 0 else 0
        print(f"{'Parameters (M)':<20} | {p_base / 1e6:<15.2f} | {p_pruned / 1e6:<15.2f} | -{impr_p:.2f}%")

        # FLOPs
        impr_f = (1 - f_pruned / f_base) * 100 if f_base > 0 else 0
        print(f"{'FLOPs (G)':<20} | {f_base / 1e9:<15.2f} | {f_pruned / 1e9:<15.2f} | -{impr_f:.2f}%")

        # FPS
        impr_fps = (fps_pruned / fps_base - 1) * 100 if fps_base > 0 else 0
        print(f"{'FPS (800x800)':<20} | {fps_base:<15.2f} | {fps_pruned:<15.2f} | +{impr_fps:.2f}%")
        print("=" * 65)

    except Exception as e:
        print(f"\nCRITICAL ERROR IN MAIN: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()