import torch
import cv2
import os
import glob
import json
import time
import re
import numpy as np
from torchvision.transforms import functional as F

# Data path
IMG_DIR = "/kaggle/input/data123/NewDeepfish/NewDeepfish/images/val"

# path Model Dense (Step 1)
DENSE_CHECKPOINT = "/kaggle/input/pth-file/pipeline2/step1_dense_det/model_best.pth"

# path Model Pruned (Step 3) & Config JSON (Step 2)
PRUNED_CHECKPOINT = "/kaggle/input/pth-file/pipeline2/step3_final_result/model_best.pth"
PRUNED_CONFIG = "/kaggle/input/pth-file/pipeline2/step2_pruned_det/backbone_lean.json"

# output path
OUTPUT_VIDEO = "comparison_demo.mp4"

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_CLASSES = 2  # Background + Fish
CONF_THRESHOLD = 0.5
FPS_VIDEO = 10  # Speed of output vid
MAX_FRAMES = 300


def load_model_custom(checkpoint_path, config_json=None):
    """Hàm load model an toàn cho Kaggle/PyTorch mới"""
    print(f"Loading: {os.path.basename(checkpoint_path)}...")

    # 1. Load Config cắt tỉa (nếu có)
    compress_rate = None
    if config_json:
        with open(config_json, 'r') as f:
            compress_rate = json.load(f)
            print(" -> Loaded Pruning Config.")

    # 2. Dựng khung Model
    model = fasterrcnn_resnet50_fpn(
        num_classes=NUM_CLASSES,
        compress_rate=compress_rate,
        weights_backbone=None
    )

    # 3. Load Weights (Fix lỗi weights_only)
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location='cpu')

    if 'model' in ckpt: ckpt = ckpt['model']

    model.load_state_dict(ckpt, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


def run_inference(model, image_tensor, original_image):
    """Chạy 1 frame và trả về ảnh đã vẽ box + thời gian xử lý"""
    t0 = time.time()
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    t1 = time.time()

    inference_time = t1 - t0
    fps_instant = 1.0 / inference_time if inference_time > 0 else 0

    # Vẽ box lên bản sao của ảnh gốc
    result_img = original_image.copy()

    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    # Lọc threshold
    keep = scores > CONF_THRESHOLD
    boxes = boxes[keep]
    scores = scores[keep]

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        # Màu xanh lá
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result_img, f"{score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return result_img, fps_instant


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """Sắp xếp tên file chuẩn (img1, img2... img10)"""
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def main():
    # Load model
    model_dense = load_model_custom(DENSE_CHECKPOINT, None)
    model_pruned = load_model_custom(PRUNED_CHECKPOINT, PRUNED_CONFIG)


    images = glob.glob(os.path.join(IMG_DIR, "*.jpg")) + \
             glob.glob(os.path.join(IMG_DIR, "*.png"))
    images.sort(key=alphanum_key)

    if not images:
        print("ERROR: Can't find any image")
        return

    if MAX_FRAMES:
        images = images[:MAX_FRAMES]
        print(f"Processing first {len(images)} images.")

    # Take shape
    sample = cv2.imread(images[0])
    h, w, _ = sample.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS_VIDEO, (w * 2, h))


    for i, img_path in enumerate(images):
        frame = cv2.imread(img_path)
        if frame is None: continue

        # Preprocess
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(img_rgb).to(DEVICE).unsqueeze(0)

        # Dense model
        res_dense, fps_d = run_inference(model_dense, img_tensor, frame)
        cv2.putText(res_dense, f"DENSE (Original)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(res_dense, f"FPS: {fps_d:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Pruned model
        res_pruned, fps_p = run_inference(model_pruned, img_tensor, frame)
        cv2.putText(res_pruned, f"PRUNED (Compressed)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(res_pruned, f"FPS: {fps_p:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


        combined_frame = np.hstack((res_dense, res_pruned))
        out.write(combined_frame)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(images)} frames...")

    out.release()
    print(f"\nVideo saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()