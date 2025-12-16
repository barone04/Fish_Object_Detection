import argparse
import cv2
import torch
import numpy as np
import os
import json
import time
from PIL import Image
from models.faster_rcnn import fasterrcnn_resnet50_fpn
from data import presets

# Danh sách class (Sửa lại tên nếu bạn muốn hiển thị tên thật)
CLASS_NAMES = [
    "__background__",
    "Fish_1"
]


def get_args():
    parser = argparse.ArgumentParser(description="Robust Video Inference")
    parser.add_argument("--input", required=True, help="Path to input video (.flv, .mp4...)")
    parser.add_argument("--output", default="output_video.avi", help="Path to output video (.avi)")
    parser.add_argument("--weight", required=True, help="Path to model .pth")
    parser.add_argument("--compress-rate", default=None, help="Path to json config")
    parser.add_argument("--conf", default=0.5, type=float, help="Confidence threshold")
    parser.add_argument("--device", default="cuda", help="Device")
    return parser.parse_args()


def draw_boxes(frame, predictions, threshold=0.5):
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()

    # Vẽ lên bản copy để không ảnh hưởng dữ liệu gốc
    frame_draw = frame.copy()

    for box, score, label in zip(boxes, scores, labels):
        if score >= threshold:
            x1, y1, x2, y2 = box.astype(int)
            color = (0, 255, 0)  # Xanh lá

            # Vẽ Box
            cv2.rectangle(frame_draw, (x1, y1), (x2, y2), color, 2)

            # Tên class
            name_idx = int(label)
            label_text = CLASS_NAMES[name_idx] if name_idx < len(CLASS_NAMES) else f"Class {name_idx}"
            caption = f"{label_text}: {score:.2f}"

            # Vẽ nền chữ đen cho dễ đọc
            (w, h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame_draw, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(frame_draw, caption, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return frame_draw


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"--- Running Inference on {device} ---")

    # 1. Load Config Nén
    cpr = None
    if args.compress_rate:
        with open(args.compress_rate, 'r') as f:
            cpr = json.load(f)
        print("Loaded compression config.")

    # 2. Load Model
    print("Loading model...")
    model = fasterrcnn_resnet50_fpn(num_classes=14, compress_rate=cpr)

    checkpoint = torch.load(args.weight, map_location='cpu')
    if 'model' in checkpoint: checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # 3. Mở Video Input
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {args.input}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input Video: {width}x{height} @ {fps:.2f} FPS ({total_frames} frames)")

    # 4. Thiết lập Video Output (Dùng MJPG cho an toàn tuyệt đối)
    save_path = args.output
    # Tự động đổi đuôi sang .avi nếu người dùng nhập .mp4 (để khớp với MJPG)
    if not save_path.endswith('.avi'):
        save_path = os.path.splitext(save_path)[0] + '.avi'

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print("ERROR: Could not initialize VideoWriter. Try installing ffmpeg.")
        return

    print(f"Output will be saved to: {save_path}")

    # 5. Vòng lặp Inference
    transform = presets.DetectionPresetEval()
    frame_cnt = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break  # Hết video

            # --- Xử lý ảnh (Fix lỗi TypeError PIL) ---
            # OpenCV (BGR) -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Numpy -> PIL Image
            pil_img = Image.fromarray(frame_rgb)
            # PIL -> Tensor (thông qua preset)
            img_tensor, _ = transform(pil_img, None)
            img_tensor = img_tensor.to(device)
            # ----------------------------------------

            # Predict
            with torch.no_grad():
                predictions = model([img_tensor])[0]

            # Vẽ (Dùng frame gốc BGR)
            final_frame = draw_boxes(frame, predictions, threshold=args.conf)

            # Ghi ra file
            out.write(final_frame)

            frame_cnt += 1
            if frame_cnt % 50 == 0:
                print(f"Processed {frame_cnt}/{total_frames} frames...")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        out.release()
        duration = time.time() - t_start
        print(f"Done! Saved to {save_path}")
        if duration > 0:
            print(f"Average Processing Speed: {frame_cnt / duration:.2f} FPS")


if __name__ == "__main__":
    main()