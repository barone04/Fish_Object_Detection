"""
TO RUN THIS FILE:
python visualize.py \
    --input ./datasets/fish_video.mp4 \
    --weight ./output/det_pipeline2_final/model_best.pth \
    --compress-rate ./output/det_pruned/config.json
"""


import argparse
import cv2
import torch
import numpy as np
from models.faster_rcnn import fasterrcnn_resnet50_fpn
from data import presets
import time


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Model
    # Cần load compress_rate nếu model đã bị nén
    cpr = None
    if args.compress_rate:
        import json
        with open(args.compress_rate, 'r') as f:
            cpr = json.load(f)

    model = fasterrcnn_resnet50_fpn(num_classes=14, compress_rate=cpr)
    checkpoint = torch.load(args.weight, map_location='cpu')
    if 'model' in checkpoint: checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # 2. Video Capture
    cap = cv2.VideoCapture(args.input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Video Writer
    out = cv2.VideoWriter('output_inference.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    transform = presets.DetectionPresetEval()

    print(f"Processing video... (Size: {width}x{height})")

    frame_cnt = 0
    t_start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Preprocess
        img_tensor, _ = transform(frame, None)
        img_tensor = img_tensor.to(device)

        # Inference
        with torch.no_grad():
            predictions = model([img_tensor])[0]

        # Draw
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if score > args.conf:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Fish {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        frame_cnt += 1
        if frame_cnt % 50 == 0: print(f"Frame {frame_cnt} processed.")

    total_time = time.time() - t_start
    print(f"Done! Average FPS: {frame_cnt / total_time:.2f}")
    cap.release()
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--compress-rate", default=None, help="Path to json config")
    parser.add_argument("--conf", default=0.5, type=float)
    args = parser.parse_args()
    main(args)