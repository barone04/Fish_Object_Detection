import argparse
import cv2
import torch
import os
import json
import random
import torchvision.transforms as T
from PIL import Image
from models.faster_rcnn import fasterrcnn_resnet50_fpn

# Danh sách class (Sửa lại nếu cần thiết)
CLASS_NAMES = [
    "__background__",
    "Fish_1"
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

            color = (0, 255, 0)

            # Vẽ Box
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

            # Vẽ Label
            label_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f"Class {label}"
            text = f"{label_name}: {score:.2f}"

            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img_draw, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(img_draw, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return img_draw


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

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

    # Lấy Random 3 ảnh
    img_dir = os.path.join(args.data_path, "test", "images")
    if not os.path.exists(img_dir):
        img_dir = os.path.join(args.data_path, "val", "images")

    all_imgs = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    selected_imgs = random.sample(all_imgs, min(3, len(all_imgs)))

    print(f"Selected images: {selected_imgs}")

    # Inference Loop
    transform = T.Compose([T.ToTensor()])  # Convert [0, 255] -> [0.0, 1.0]

    for img_name in selected_imgs:
        img_path = os.path.join(img_dir, img_name)

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Preprocess
        img_tensor = transform(img_rgb).to(device)

        # Inference
        with torch.no_grad():
            prediction = model([img_tensor])[0]

        result_img = draw_boxes(img_bgr, prediction, threshold=args.conf)

        save_path = os.path.join(args.output_dir, f"pred_{img_name}")
        cv2.imwrite(save_path, result_img)
        print(f"Saved result to: {save_path}")

    print("result images from folder:", args.output_dir)


if __name__ == "__main__":
    main()