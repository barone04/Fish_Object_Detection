# """
# TO RUN THIS FILE:
# python visualize_image.py \
#     --input ./datasets/fish_video.mp4 \
#     --weight ./output/det_pipeline2_final/model_best.pth \
#     --compress-rate ./output/det_pruned/config.json
#
#
# """
# from PIL import Image
# import argparse
# import cv2
# import torch
# import numpy as np
# import os
# import json
# import time
# from models.faster_rcnn import fasterrcnn_resnet50_fpn
# from data import presets
#
# # --- CẤU HÌNH TÊN LOÀI CÁ (Sửa lại cho đúng dataset của bạn) ---
# # Dataset của bạn có 13 loài cá (theo thứ tự ID 1 -> 13)
# # ID 0 luôn là Background
# CLASS_NAMES = [
#     "__background__",
#     "Fish_1", "Fish_2", "Fish_3", "Fish_4", "Fish_5",
#     "Fish_6", "Fish_7", "Fish_8", "Fish_9", "Fish_10",
#     "Fish_11", "Fish_12", "Fish_13"
# ]
#
#
# # Nếu bạn biết tên thật (ví dụ: Tuna, Salmon...) hãy thay thế vào list trên.
#
# def get_args():
#     parser = argparse.ArgumentParser(description="Video Inference")
#     parser.add_argument("--input", required=True, help="Path to input video (e.g., test.mp4)")
#     parser.add_argument("--output", default="output_video.mp4", help="Path to output video")
#     parser.add_argument("--weight", required=True, help="Path to trained model (.pth)")
#     parser.add_argument("--compress-rate", default=None, help="Path to json config (if model is pruned)")
#     parser.add_argument("--conf", default=0.5, type=float, help="Confidence threshold (0.0 - 1.0)")
#     parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
#     return parser.parse_args()
#
#
# def draw_boxes(frame, predictions, threshold=0.5):
#     # Lấy dữ liệu từ GPU về CPU
#     boxes = predictions['boxes'].cpu().numpy()
#     scores = predictions['scores'].cpu().numpy()
#     labels = predictions['labels'].cpu().numpy()
#
#     for box, score, label in zip(boxes, scores, labels):
#         if score >= threshold:
#             x1, y1, x2, y2 = box.astype(int)
#
#             # Chọn màu ngẫu nhiên hoặc cố định theo class
#             color = (0, 255, 0)  # Màu xanh lá
#
#             # Vẽ Box
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#
#             # Vẽ Label
#             if label < len(CLASS_NAMES):
#                 class_name = CLASS_NAMES[label]
#             else:
#                 class_name = f"Class {label}"
#
#             text = f"{class_name}: {score:.2f}"
#             cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#
#     return frame
#
#
# def main():
#     args = get_args()
#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     # 1. Load Config Nén (nếu có)
#     cpr = None
#     if args.compress_rate:
#         with open(args.compress_rate, 'r') as f:
#             cpr = json.load(f)
#         print("Loaded compress rate config.")
#
#     # 2. Khởi tạo Model
#     # Lưu ý: num_classes phải khớp với lúc train (14 = 13 cá + 1 nền)
#     print("Creating model...")
#     model = fasterrcnn_resnet50_fpn(num_classes=14, compress_rate=cpr)
#
#     # 3. Load Trọng số
#     print(f"Loading weights from {args.weight}...")
#     checkpoint = torch.load(args.weight, map_location='cpu')
#     if 'model' in checkpoint:
#         checkpoint = checkpoint['model']
#     model.load_state_dict(checkpoint)
#     model.to(device)
#     model.eval()
#
#     # 4. Xử lý Video
#     cap = cv2.VideoCapture(args.input)
#     if not cap.isOpened():
#         print(f"Error: Cannot open video {args.input}")
#         return
#
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     # Video Writer
#     # fourcc = cv2.VideoWriter_fourcc(*'avc1')
#     # out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
#
#     # --- SỬA THÀNH ---
#     # Dùng codec MJPG và đổi đuôi file thành .avi
#     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#     output_filename = args.output
#     if output_filename.endswith('.mp4'):
#         output_filename = output_filename.replace('.mp4', '.avi')  # Tự động đổi đuôi
#
#     out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
#
#     print(f"Processing video: {width}x{height} @ {fps} FPS ({total_frames} frames)")
#
#     transform = presets.DetectionPresetEval()
#
#     frame_idx = 0
#     t_start_total = time.time()
#     inference_times = []
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret: break
#
#         # 1. OpenCV đọc ảnh dạng BGR -> Convert sang RGB
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # 2. Convert sang PIL Image (Để khớp với DetectionPresetEval)
#         img_pil = Image.fromarray(frame_rgb)
#
#         # 3. Apply transforms (PIL -> Tensor -> Float -> Normalize nếu có)
#         # transform trả về (image, target), ta chỉ cần image
#         img_tensor, _ = transform(img_pil, None)
#
#         # 4. Đẩy lên GPU
#         img_tensor = img_tensor.to(device)
#
#         # Inference
#         t0 = time.time()
#         with torch.no_grad():
#             predictions = model([img_tensor])[0]
#         t1 = time.time()
#         inference_times.append(t1 - t0)
#
#         # Vẽ hình
#         out_frame = draw_boxes(frame, predictions, threshold=args.conf)
#
#         out.write(out_frame)
#
#         frame_idx += 1
#         if frame_idx % 50 == 0:
#             avg_fps = 1.0 / np.mean(inference_times[-50:])
#             print(f"Frame {frame_idx}/{total_frames} - FPS: {avg_fps:.2f}")
#
#     cap.release()
#     out.release()
#
#     total_duration = time.time() - t_start_total
#     avg_inference_fps = 1.0 / np.mean(inference_times)
#     print("=======================================")
#     print(f"Done! Video saved to: {args.output}")
#     print(f"Total time: {total_duration:.2f}s")
#     print(f"Average Inference FPS: {avg_inference_fps:.2f}")
#     print("=======================================")
#
#
# if __name__ == "__main__":
#     main()

import argparse
import cv2
import torch
import numpy as np
import os
import json
import random
import torchvision.transforms as T
from PIL import Image
from models.faster_rcnn import fasterrcnn_resnet50_fpn

# Danh sách class (Sửa lại nếu cần thiết)
CLASS_NAMES = [
    "__background__",
    "Fish_1", "Fish_2", "Fish_3", "Fish_4", "Fish_5",
    "Fish_6", "Fish_7", "Fish_8", "Fish_9", "Fish_10",
    "Fish_11", "Fish_12", "Fish_13"
]


def get_args():
    parser = argparse.ArgumentParser(description="Image Inference")
    parser.add_argument("--data-path", required=True, help="Root path to dataset (e.g., ./datasets/Fish_Dataset)")
    parser.add_argument("--weight", required=True, help="Path to model .pth")
    parser.add_argument("--compress-rate", default=None, help="Path to json config")
    parser.add_argument("--output-dir", default="./output_images", help="Folder to save results")
    parser.add_argument("--conf", default=0.5, type=float, help="Confidence threshold")
    parser.add_argument("--device", default="cuda", help="Device")
    return parser.parse_args()


def draw_boxes(img_bgr, predictions, threshold=0.5):
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()

    img_draw = img_bgr.copy()

    for box, score, label in zip(boxes, scores, labels):
        if score >= threshold:
            x1, y1, x2, y2 = box.astype(int)

            # Màu xanh lá
            color = (0, 255, 0)

            # Vẽ Box
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

            # Vẽ Label
            label_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f"Class {label}"
            text = f"{label_name}: {score:.2f}"

            # Vẽ nền chữ cho dễ đọc
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img_draw, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(img_draw, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return img_draw


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Tạo thư mục output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 2. Load Config & Model
    cpr = None
    if args.compress_rate:
        with open(args.compress_rate, 'r') as f:
            cpr = json.load(f)

    print("Creating model...")
    model = fasterrcnn_resnet50_fpn(num_classes=14, compress_rate=cpr)

    print(f"Loading weights from {args.weight}...")
    checkpoint = torch.load(args.weight, map_location='cpu')
    if 'model' in checkpoint: checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # 3. Lấy Random 3 ảnh
    # Giả sử cấu trúc: data_path/val/images/*.png hoặc *.jpg
    img_dir = os.path.join(args.data_path, "test", "images")
    if not os.path.exists(img_dir):
        # Fallback thử folder train nếu val không có
        img_dir = os.path.join(args.data_path, "val", "images")

    all_imgs = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    selected_imgs = random.sample(all_imgs, min(3, len(all_imgs)))

    print(f"Selected images: {selected_imgs}")

    # 4. Inference Loop
    transform = T.Compose([T.ToTensor()])  # Convert [0, 255] -> [0.0, 1.0]

    for img_name in selected_imgs:
        img_path = os.path.join(img_dir, img_name)

        # Đọc ảnh bằng OpenCV (BGR)
        img_bgr = cv2.imread(img_path)
        # Convert sang RGB để đưa vào model
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Preprocess
        img_tensor = transform(img_rgb).to(device)

        # Inference
        with torch.no_grad():
            prediction = model([img_tensor])[0]

        # Vẽ kết quả lên ảnh gốc (BGR)
        result_img = draw_boxes(img_bgr, prediction, threshold=args.conf)

        # Lưu ảnh
        save_path = os.path.join(args.output_dir, f"pred_{img_name}")
        cv2.imwrite(save_path, result_img)
        print(f"Saved result to: {save_path}")

    print("Done! Please download the images from folder:", args.output_dir)


if __name__ == "__main__":
    main()